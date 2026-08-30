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
from .model import ConditionalGAN, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"


def _maybe_save_image_grid(images: torch.Tensor, path: str | Path) -> None:
    from dlhub.artifacts import save_image_if_available

    save_image_if_available(images, path, nrow=8)


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 09 (Generative): compact class-conditional GAN on synthetic images."
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)

    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)

    args = parser.parse_args()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
    )
    data_cfg = DataConfig(
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_classes=args.num_classes,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        seed=args.data_seed,
        val_fraction=args.val_fraction,
    )
    model_cfg = ModelConfig(
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        image_size=args.image_size,
    )
    return train_cfg, data_cfg, model_cfg


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="generative",
        lesson="lesson_09_compact_conditional_gan",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("generative.compact_conditional_gan", log_file=paths.logs_dir / "train.log")
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
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    train_loader, val_loader = get_dataloaders(data_cfg)
    bundle = ConditionalGAN(model_cfg).to(device_info.torch_device)
    gen = bundle.generator
    disc = bundle.discriminator
    bce = torch.nn.BCEWithLogitsLoss()

    opt_g = torch.optim.Adam(
        gen.parameters(), lr=train_cfg.learning_rate, betas=(train_cfg.beta1, train_cfg.beta2)
    )
    opt_d = torch.optim.Adam(
        disc.parameters(), lr=train_cfg.learning_rate, betas=(train_cfg.beta1, train_cfg.beta2)
    )

    fixed_z = torch.randn(64, model_cfg.z_dim, device=device_info.torch_device)
    fixed_labels = torch.arange(64, device=device_info.torch_device, dtype=torch.long) % int(
        model_cfg.num_classes
    )
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, train_cfg.epochs + 1):
        gen.train()
        disc.train()
        d_loss_total = 0.0
        g_loss_total = 0.0
        seen = 0

        for batch_idx, (real_images, labels) in enumerate(train_loader):
            if train_cfg.max_train_batches is not None and batch_idx >= train_cfg.max_train_batches:
                break

            real_images = real_images.to(device_info.torch_device)
            labels = labels.to(device_info.torch_device).long()
            bsz = int(real_images.size(0))
            seen += bsz

            real_targets = torch.ones(bsz, device=device_info.torch_device)
            fake_targets = torch.zeros(bsz, device=device_info.torch_device)

            z = torch.randn(bsz, model_cfg.z_dim, device=device_info.torch_device)
            fake_images = gen(z, labels).detach()

            opt_d.zero_grad(set_to_none=True)
            d_real = disc(real_images, labels)
            d_fake = disc(fake_images, labels)
            d_loss = bce(d_real, real_targets) + bce(d_fake, fake_targets)
            d_loss.backward()
            opt_d.step()

            z = torch.randn(bsz, model_cfg.z_dim, device=device_info.torch_device)
            opt_g.zero_grad(set_to_none=True)
            fake_images = gen(z, labels)
            g_loss = bce(disc(fake_images, labels), real_targets)
            g_loss.backward()
            opt_g.step()

            d_loss_total += float(d_loss.item()) * bsz
            g_loss_total += float(g_loss.item()) * bsz

        d_loss_avg = d_loss_total / max(1, seen)
        g_loss_avg = g_loss_total / max(1, seen)

        gen.eval()
        disc.eval()
        val_d_loss_total = 0.0
        val_g_loss_total = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch_idx, (real_images, labels) in enumerate(val_loader):
                if (
                    train_cfg.max_eval_batches is not None
                    and batch_idx >= train_cfg.max_eval_batches
                ):
                    break

                real_images = real_images.to(device_info.torch_device)
                labels = labels.to(device_info.torch_device).long()
                bsz = int(real_images.size(0))
                val_seen += bsz

                real_targets = torch.ones(bsz, device=device_info.torch_device)
                fake_targets = torch.zeros(bsz, device=device_info.torch_device)
                z = torch.randn(bsz, model_cfg.z_dim, device=device_info.torch_device)
                fake_images = gen(z, labels)
                val_d_loss = bce(disc(real_images, labels), real_targets) + bce(
                    disc(fake_images, labels), fake_targets
                )
                val_g_loss = bce(disc(fake_images, labels), real_targets)
                val_d_loss_total += float(val_d_loss.item()) * bsz
                val_g_loss_total += float(val_g_loss.item()) * bsz

        val_d_loss_avg = val_d_loss_total / max(1, val_seen)
        val_g_loss_avg = val_g_loss_total / max(1, val_seen)
        logger.info(
            "Epoch %d/%d | d_loss %.4f | g_loss %.4f | val_d_loss %.4f | val_g_loss %.4f",
            epoch,
            train_cfg.epochs,
            d_loss_avg,
            g_loss_avg,
            val_d_loss_avg,
            val_g_loss_avg,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "d_loss": d_loss_avg,
                "g_loss": g_loss_avg,
                "val_d_loss": val_d_loss_avg,
                "val_g_loss": val_g_loss_avg,
            },
        )

        with torch.no_grad():
            samples = gen(fixed_z, fixed_labels).cpu()
        torch.save({"samples": samples, "labels": fixed_labels.cpu()}, paths.run_dir / "samples.pt")
        _maybe_save_image_grid(samples, paths.run_dir / "samples.png")

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=bundle,
        optimizer=None,
        epoch=train_cfg.epochs,
        extra={"track": "generative", "lesson": "lesson_09_compact_conditional_gan"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.generative.lesson_09_compact_conditional_gan.train"
        )
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
