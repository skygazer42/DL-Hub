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

from .data import DataConfig, get_dataloaders
from .model import ModelConfig, VAE, vae_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    beta: float = 1.0
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"


def _maybe_save_image_grid(images: torch.Tensor, path: str | Path) -> None:
    try:
        from torchvision.utils import save_image
    except Exception:
        return
    save_image(images, path, nrow=8)


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(description="Lesson 01 (Generative): MLP VAE on MNIST (or fake).")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)

    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=400)

    parser.add_argument("--dataset", type=str, default="fake", choices=["fake", "mnist"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)

    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
    )
    data_cfg = DataConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        seed=args.data_seed,
        val_fraction=args.val_fraction,
    )
    model_cfg = ModelConfig(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
    return train_cfg, data_cfg, model_cfg


def _flatten_images(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim != 4 or batch.shape[1:] != (1, 28, 28):
        raise ValueError(f"Expected batch shape (B, 1, 28, 28), got {tuple(batch.shape)}")
    return batch.view(batch.size(0), -1)


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(track="generative", lesson="lesson_01_vae_mnist", run_name=train_cfg.run_name)
    logger = get_logger("generative.vae_mnist", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Dataset: %s", data_cfg.dataset)
    logger.info("Outputs: %s", paths.run_dir)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "model": dataclass_to_dict(model_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    train_loader, val_loader = get_dataloaders(data_cfg)
    model = VAE(model_cfg).to(device_info.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    fixed_z = torch.randn(64, model_cfg.latent_dim, device=device_info.torch_device)
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_seen = 0

        for batch_idx, images in enumerate(train_loader):
            if train_cfg.max_train_batches is not None and batch_idx >= train_cfg.max_train_batches:
                break

            images = images.to(device_info.torch_device)
            x = _flatten_images(images)

            optimizer.zero_grad(set_to_none=True)
            recon_logits, mu, logvar = model(x)
            loss, recon, kl = vae_loss(
                recon_logits=recon_logits, x=x, mu=mu, logvar=logvar, beta=train_cfg.beta
            )
            loss.backward()
            optimizer.step()

            bsz = int(images.shape[0])
            total_seen += bsz
            total_loss += float(loss.item()) * bsz
            total_recon += float(recon.item()) * bsz
            total_kl += float(kl.item()) * bsz

        train_loss = total_loss / max(1, total_seen)
        train_recon = total_recon / max(1, total_seen)
        train_kl = total_kl / max(1, total_seen)

        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch_idx, images in enumerate(val_loader):
                if train_cfg.max_eval_batches is not None and batch_idx >= train_cfg.max_eval_batches:
                    break
                images = images.to(device_info.torch_device)
                x = _flatten_images(images)
                recon_logits, mu, logvar = model(x)
                loss, _, _ = vae_loss(
                    recon_logits=recon_logits, x=x, mu=mu, logvar=logvar, beta=train_cfg.beta
                )
                bsz = int(images.shape[0])
                val_seen += bsz
                val_loss += float(loss.item()) * bsz

        val_loss = val_loss / max(1, val_seen)

        logger.info(
            "Epoch %d/%d | train loss %.4f (recon %.4f, kl %.4f) | val loss %.4f",
            epoch,
            train_cfg.epochs,
            train_loss,
            train_recon,
            train_kl,
            val_loss,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_recon": train_recon,
                "train_kl": train_kl,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

        with torch.no_grad():
            sample_logits = model.decode(fixed_z)
            samples = torch.sigmoid(sample_logits).view(-1, 1, 28, 28).cpu()
        torch.save({"samples": samples}, paths.run_dir / "samples.pt")
        _maybe_save_image_grid(samples, paths.run_dir / "samples.png")

    # Save reconstructions for one batch (inputs + reconstructions).
    model.eval()
    with torch.no_grad():
        images = next(iter(val_loader)).to(device_info.torch_device)
        x = _flatten_images(images)
        recon_logits, _, _ = model(x)
        recon = torch.sigmoid(recon_logits).view(-1, 1, 28, 28).cpu()
        inp = images.cpu()
    torch.save({"inputs": inp, "reconstructions": recon}, paths.run_dir / "recons.pt")
    _maybe_save_image_grid(torch.cat([inp[:32], recon[:32]], dim=0), paths.run_dir / "recons.png")

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=train_cfg.epochs,
        extra={"track": "generative", "lesson": "lesson_01_vae_mnist"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.generative.lesson_01_vae_mnist.train"
        )
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
