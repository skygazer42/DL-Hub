
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

from .data import DataConfig, get_dataloader
from .model import GAN, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    run_name: str = "dev"

    label_smoothing: float = 0.0  # e.g. 0.1 => real label becomes 0.9


def _maybe_save_image_grid(images: torch.Tensor, path: str | Path) -> None:
    try:
        from torchvision.utils import save_image
    except Exception:
        return
    save_image(images, path, nrow=8)


def parse_args() -> tuple[TrainConfig, DataConfig, ModelConfig]:
    parser = argparse.ArgumentParser(description="Lesson 02 (Generative): Vanilla GAN on MNIST (or fake).")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument("--dataset", type=str, default="fake", choices=["fake", "mnist"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--data-seed", type=int, default=0)

    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        run_name=args.run_name,
        label_smoothing=args.label_smoothing,
    )
    data_cfg = DataConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        seed=args.data_seed,
    )
    model_cfg = ModelConfig(z_dim=args.z_dim, hidden_dim=args.hidden_dim)
    return train_cfg, data_cfg, model_cfg


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(track="generative", lesson="lesson_02_gan_mnist", run_name=train_cfg.run_name)
    logger = get_logger("generative.gan_mnist", log_file=paths.logs_dir / "train.log")
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

    loader = get_dataloader(data_cfg)
    bundle = GAN(model_cfg).to(device_info.torch_device)
    gen = bundle.generator
    disc = bundle.discriminator

    opt_g = torch.optim.Adam(
        gen.parameters(), lr=train_cfg.learning_rate, betas=(train_cfg.beta1, train_cfg.beta2)
    )
    opt_d = torch.optim.Adam(
        disc.parameters(), lr=train_cfg.learning_rate, betas=(train_cfg.beta1, train_cfg.beta2)
    )
    bce = torch.nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, model_cfg.z_dim, device=device_info.torch_device)
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, train_cfg.epochs + 1):
        gen.train()
        disc.train()

        d_loss_total = 0.0
        g_loss_total = 0.0
        seen = 0

        for batch_idx, real_images in enumerate(loader):
            if train_cfg.max_train_batches is not None and batch_idx >= train_cfg.max_train_batches:
                break

            real_images = real_images.to(device_info.torch_device)
            bsz = int(real_images.size(0))
            seen += bsz

            real_label = torch.ones(bsz, device=device_info.torch_device) * (1.0 - train_cfg.label_smoothing)
            fake_label = torch.zeros(bsz, device=device_info.torch_device)

            # 1) Train discriminator.
            z = torch.randn(bsz, model_cfg.z_dim, device=device_info.torch_device)
            fake_images = gen(z).detach()

            opt_d.zero_grad(set_to_none=True)
            logits_real = disc(real_images)
            logits_fake = disc(fake_images)
            loss_real = bce(logits_real, real_label)
            loss_fake = bce(logits_fake, fake_label)
            d_loss = loss_real + loss_fake
            d_loss.backward()
            opt_d.step()

            # 2) Train generator.
            z = torch.randn(bsz, model_cfg.z_dim, device=device_info.torch_device)
            fake_images = gen(z)
            opt_g.zero_grad(set_to_none=True)
            logits_fake_for_g = disc(fake_images)
            g_loss = bce(logits_fake_for_g, real_label)
            g_loss.backward()
            opt_g.step()

            d_loss_total += float(d_loss.item()) * bsz
            g_loss_total += float(g_loss.item()) * bsz

        d_loss_avg = d_loss_total / max(1, seen)
        g_loss_avg = g_loss_total / max(1, seen)

        logger.info("Epoch %d/%d | d_loss %.4f | g_loss %.4f", epoch, train_cfg.epochs, d_loss_avg, g_loss_avg)
        append_jsonl(metrics_path, {"epoch": epoch, "d_loss": d_loss_avg, "g_loss": g_loss_avg})

        with torch.no_grad():
            samples = gen(fixed_z).cpu()
        torch.save({"samples": samples}, paths.run_dir / "samples.pt")
        _maybe_save_image_grid(samples, paths.run_dir / "samples.png")

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=bundle,
        optimizer=None,  # GAN uses two optimizers; keep checkpoint lightweight.
        epoch=train_cfg.epochs,
        extra={"track": "generative", "lesson": "lesson_02_gan_mnist"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.generative.lesson_02_gan_mnist.train"
        )
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
