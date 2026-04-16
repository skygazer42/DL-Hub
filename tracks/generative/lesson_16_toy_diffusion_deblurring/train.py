import argparse
import sys
from dataclasses import dataclass

import torch

from dlhub.checkpoint import save_checkpoint
from dlhub.config import append_jsonl, dataclass_to_dict, write_json
from dlhub.device import resolve_device
from dlhub.logging import get_logger
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed

from .data import DataConfig, get_dataloaders
from .model import DiffusionSchedule, ModelConfig, ToyDeblurringDiffusionModel, q_sample


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"
    num_sample_steps: int | None = None


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig, DiffusionSchedule]:
    parser = argparse.ArgumentParser(
        description="Lesson 16 (Generative): toy diffusion-style image deblurring from blurry/sharp pairs."
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--num-sample-steps", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)

    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--time-embed-dim", type=int, default=32)

    parser.add_argument("--num-diffusion-steps", type=int, default=20)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)

    args = parser.parse_args()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
        num_sample_steps=args.num_sample_steps,
    )
    data_cfg = DataConfig(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        seed=args.data_seed,
        val_fraction=args.val_fraction,
    )
    model_cfg = ModelConfig(
        image_size=args.image_size,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        time_embed_dim=args.time_embed_dim,
    )
    schedule = DiffusionSchedule(
        num_steps=args.num_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    return train_cfg, data_cfg, model_cfg, schedule


def _evaluate(
    model: ToyDeblurringDiffusionModel,
    loader: torch.utils.data.DataLoader,
    schedule: DiffusionSchedule,
    *,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    with torch.no_grad():
        for batch_idx, (blurry, sharp) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            blurry = blurry.to(device)
            sharp = sharp.to(device)
            noise = torch.randn_like(sharp)
            timesteps = torch.randint(0, schedule.num_steps, (sharp.size(0),), device=device)
            xt = q_sample(schedule, sharp, timesteps, noise)
            pred_noise = model(xt=xt, blurry=blurry, timesteps=timesteps)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)

            bsz = int(sharp.size(0))
            total_loss += float(loss.item()) * bsz
            total_seen += bsz
    return total_loss / max(1, total_seen)


def run_training(
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    schedule: DiffusionSchedule,
) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="generative",
        lesson="lesson_16_toy_diffusion_deblurring",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("generative.toy_diffusion_deblurring", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "model": dataclass_to_dict(model_cfg),
            "schedule": dataclass_to_dict(schedule),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = ToyDeblurringDiffusionModel(model_cfg).to(device_info.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for batch_idx, (blurry, sharp) in enumerate(train_loader):
            if train_cfg.max_train_batches is not None and batch_idx >= train_cfg.max_train_batches:
                break
            blurry = blurry.to(device_info.torch_device)
            sharp = sharp.to(device_info.torch_device)
            noise = torch.randn_like(sharp)
            timesteps = torch.randint(0, schedule.num_steps, (sharp.size(0),), device=device_info.torch_device)
            xt = q_sample(schedule, sharp, timesteps, noise)

            optimizer.zero_grad(set_to_none=True)
            pred_noise = model(xt=xt, blurry=blurry, timesteps=timesteps)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()

            bsz = int(sharp.size(0))
            total_loss += float(loss.item()) * bsz
            total_seen += bsz

        train_loss = total_loss / max(1, total_seen)
        val_loss = _evaluate(
            model,
            val_loader,
            schedule,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
        )
        logger.info(
            "Epoch %d/%d | train_noise_mse %.4f | val_noise_mse %.4f",
            epoch,
            train_cfg.epochs,
            train_loss,
            val_loss,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_noise_mse": train_loss,
                "val_noise_mse": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    sample_blurry, sample_sharp = next(iter(val_loader))
    sample_blurry = sample_blurry[:16]
    sample_sharp = sample_sharp[:16]
    samples = model.sample(
        schedule=schedule,
        blurry=sample_blurry,
        device=device_info.torch_device,
        num_steps=train_cfg.num_sample_steps,
    )
    denoise_frames = model.sample(
        schedule=schedule,
        blurry=sample_blurry,
        device=device_info.torch_device,
        num_steps=train_cfg.num_sample_steps,
        return_all=True,
    )
    torch.save(
        {
            "blurry": sample_blurry,
            "sharp": sample_sharp,
            "samples": samples,
        },
        paths.run_dir / "samples.pt",
    )
    torch.save({"frames": denoise_frames}, paths.run_dir / "denoise_grid.pt")

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=train_cfg.epochs,
        extra={"track": "generative", "lesson": "lesson_16_toy_diffusion_deblurring"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.generative.lesson_16_toy_diffusion_deblurring.train"
        )
    train_cfg, data_cfg, model_cfg, schedule = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg, schedule)


if __name__ == "__main__":
    raise SystemExit(main())
