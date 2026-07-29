from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from dlhub.checkpoint import save_checkpoint
from dlhub.config import append_jsonl, dataclass_to_dict, write_json
from dlhub.device import resolve_device
from dlhub.logging import get_logger
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed
from dlhub.vision.super_resolution import compute_psnr

from .data import DataConfig, get_dataloaders
from .model import ModelConfig, build_model


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"

    arch: str = "sr:srcnn_tiny"
    in_channels: int = 3
    upscale_factor: int = 2
    width_mult: float = 1.0
    dropout: float = 0.0


def _maybe_save_image(image: torch.Tensor, path: str | Path) -> None:
    from dlhub.artifacts import save_image_if_available

    save_image_if_available(image, path)


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 17 (Vision): synthetic paired super-resolution."
    )
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--upscale-factor", type=int, default=2)
    parser.add_argument("--blur-kernel-size", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--arch", type=str, default="sr:srcnn_tiny")
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        arch=args.arch,
        in_channels=args.in_channels,
        upscale_factor=args.upscale_factor,
        width_mult=args.width_mult,
        dropout=args.dropout,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        in_channels=args.in_channels,
        upscale_factor=args.upscale_factor,
        blur_kernel_size=args.blur_kernel_size,
        noise_std=args.noise_std,
    )
    return train_cfg, data_cfg


def _forward_batch(
    model: torch.nn.Module,
    *,
    low_res: torch.Tensor,
    high_res: torch.Tensor,
    criterion: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(low_res)
    if not isinstance(out, dict) or "sr" not in out:
        raise RuntimeError(f"Expected model output dict containing 'sr', got {type(out).__name__}")
    sr = out["sr"]
    loss = criterion(sr, high_res)
    psnr = compute_psnr(sr.detach(), high_res.detach())
    return loss, psnr


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    criterion: torch.nn.Module,
    max_batches: int | None,
) -> tuple[float, float, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_psnr = 0.0
    steps = 0
    preview = None

    for batch_idx, (low_res, high_res) in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        low_res = low_res.to(device)
        high_res = high_res.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        if is_train:
            loss, psnr = _forward_batch(
                model,
                low_res=low_res,
                high_res=high_res,
                criterion=criterion,
            )
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss, psnr = _forward_batch(
                    model,
                    low_res=low_res,
                    high_res=high_res,
                    criterion=criterion,
                )

        total_loss += float(loss.detach().item())
        total_psnr += float(psnr.detach().item())
        steps += 1

        if preview is None:
            with torch.no_grad():
                sr = model(low_res)["sr"].detach().cpu()
            preview = (low_res.detach().cpu(), high_res.detach().cpu(), sr)

    if steps == 0:
        raise RuntimeError("No batches were processed. Check dataset size or max_batches.")
    return total_loss / steps, total_psnr / steps, preview


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)
    paths = build_run_paths(
        track="vision",
        lesson="lesson_17_synthetic_super_resolution",
        run_name=train_cfg.run_name,
    )
    logger = get_logger(
        "vision.synthetic_super_resolution",
        log_file=paths.logs_dir / "train.log",
    )
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Arch: %s", train_cfg.arch)
    logger.info("Outputs: %s", paths.run_dir)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = build_model(
        ModelConfig(
            arch=train_cfg.arch,
            in_channels=train_cfg.in_channels,
            upscale_factor=train_cfg.upscale_factor,
            image_size=data_cfg.image_size,
            width_mult=train_cfg.width_mult,
            dropout=train_cfg.dropout,
        )
    ).to(device_info.torch_device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))

    metrics_path = paths.run_dir / "metrics.jsonl"
    last_preview = None
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_loss, train_psnr, _ = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device_info.torch_device,
            criterion=criterion,
            max_batches=train_cfg.max_train_batches,
        )
        eval_loss, eval_psnr, last_preview = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device_info.torch_device,
            criterion=criterion,
            max_batches=train_cfg.max_eval_batches,
        )
        logger.info(
            "Epoch %d/%d | train l1 %.4f psnr %.2f | eval l1 %.4f psnr %.2f",
            epoch,
            train_cfg.epochs,
            train_loss,
            train_psnr,
            eval_loss,
            eval_psnr,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_l1": train_loss,
                "train_psnr": train_psnr,
                "eval_l1": eval_loss,
                "eval_psnr": eval_psnr,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_17_synthetic_super_resolution"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)

    if last_preview is not None:
        low_res, high_res, sr = last_preview
        torch.save(
            {"low_res": low_res, "high_res": high_res, "sr": sr},
            paths.run_dir / "predictions.pt",
        )
        _maybe_save_image(sr[:1], paths.run_dir / "preview.png")
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_17_synthetic_super_resolution.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
