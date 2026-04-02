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
from dlhub.vision.style_transfer_zoo import build_local_model

from .data import DataConfig, get_dataloaders


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    run_name: str = "dev"

    arch: str = "dlst:cyclegan_tiny"
    width_mult: float = 1.0
    dropout: float = 0.0

    lambda_gan: float = 1.0
    lambda_cycle: float = 10.0


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(
        description="Lesson 16 (Vision): Translation style transfer (CycleGAN-style, toy-first)."
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--arch", type=str, default="dlst:cyclegan_tiny")
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-gan", type=float, default=1.0)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)

    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)

    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        run_name=args.run_name,
        arch=args.arch,
        width_mult=args.width_mult,
        dropout=args.dropout,
        lambda_gan=args.lambda_gan,
        lambda_cycle=args.lambda_cycle,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.data_seed,
        num_workers=args.num_workers,
    )
    return train_cfg, data_cfg


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="vision",
        lesson="lesson_16_style_transfer_translation_cyclegan",
        run_name=train_cfg.run_name,
    )
    logger = get_logger("vision.style_transfer_cyclegan", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    train_loader, _ = get_dataloaders(data_cfg)
    model = build_local_model(
        train_cfg.arch,
        in_channels=int(data_cfg.in_channels),
        image_size=int(data_cfg.image_size),
        width_mult=float(train_cfg.width_mult),
        dropout=float(train_cfg.dropout),
    ).to(device_info.torch_device)

    # The local CycleGAN model exposes these submodules.
    g_params = list(model.g_ab.parameters()) + list(model.g_ba.parameters())
    d_params = list(model.d_a.parameters()) + list(model.d_b.parameters())

    opt_g = torch.optim.Adam(g_params, lr=float(train_cfg.learning_rate), betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d_params, lr=float(train_cfg.learning_rate), betas=(0.5, 0.999))

    bce = torch.nn.BCEWithLogitsLoss()
    l1 = torch.nn.L1Loss()
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, int(train_cfg.epochs) + 1):
        model.train()
        d_loss_total = 0.0
        g_loss_total = 0.0
        seen = 0

        for batch_idx, (a, b) in enumerate(train_loader):
            if train_cfg.max_train_batches is not None and batch_idx >= int(train_cfg.max_train_batches):
                break
            a = a.to(device_info.torch_device)
            b = b.to(device_info.torch_device)
            bsz = int(a.size(0))
            seen += bsz

            ones_a = torch.ones(bsz, 1, device=device_info.torch_device)
            zeros_a = torch.zeros(bsz, 1, device=device_info.torch_device)

            # 1) Discriminators.
            with torch.no_grad():
                fake_b = model.g_ab(a)
                fake_a = model.g_ba(b)

            opt_d.zero_grad(set_to_none=True)
            logits_a_real = model.d_a(a).mean(dim=(2, 3))
            logits_a_fake = model.d_a(fake_a.detach()).mean(dim=(2, 3))
            logits_b_real = model.d_b(b).mean(dim=(2, 3))
            logits_b_fake = model.d_b(fake_b.detach()).mean(dim=(2, 3))

            d_loss = (
                bce(logits_a_real, ones_a)
                + bce(logits_a_fake, zeros_a)
                + bce(logits_b_real, ones_a)
                + bce(logits_b_fake, zeros_a)
            )
            d_loss.backward()
            opt_d.step()

            # 2) Generators.
            opt_g.zero_grad(set_to_none=True)
            fake_b = model.g_ab(a)
            fake_a = model.g_ba(b)
            rec_a = model.g_ba(fake_b)
            rec_b = model.g_ab(fake_a)

            logits_b_fake_for_g = model.d_b(fake_b).mean(dim=(2, 3))
            logits_a_fake_for_g = model.d_a(fake_a).mean(dim=(2, 3))

            gan_loss = bce(logits_b_fake_for_g, ones_a) + bce(logits_a_fake_for_g, ones_a)
            cycle_loss = l1(rec_a, a) + l1(rec_b, b)
            g_loss = float(train_cfg.lambda_gan) * gan_loss + float(train_cfg.lambda_cycle) * cycle_loss
            g_loss.backward()
            opt_g.step()

            d_loss_total += float(d_loss.detach().item()) * bsz
            g_loss_total += float(g_loss.detach().item()) * bsz

        d_loss_avg = d_loss_total / max(1, seen)
        g_loss_avg = g_loss_total / max(1, seen)
        logger.info("Epoch %d/%d | d_loss %.4f | g_loss %.4f", epoch, train_cfg.epochs, d_loss_avg, g_loss_avg)
        append_jsonl(metrics_path, {"epoch": epoch, "d_loss": d_loss_avg, "g_loss": g_loss_avg})

        # Save a small sample pack for inspection.
        model.eval()
        with torch.no_grad():
            a_s, b_s = next(iter(train_loader))
            a_s = a_s[:4].to(device_info.torch_device)
            b_s = b_s[:4].to(device_info.torch_device)
            fake_b_s = model.g_ab(a_s)
            fake_a_s = model.g_ba(b_s)
            rec_a_s = model.g_ba(fake_b_s)
            rec_b_s = model.g_ab(fake_a_s)
        torch.save(
            {
                "a": a_s.cpu(),
                "b": b_s.cpu(),
                "fake_b": fake_b_s.cpu(),
                "fake_a": fake_a_s.cpu(),
                "rec_a": rec_a_s.cpu(),
                "rec_b": rec_b_s.cpu(),
            },
            paths.run_dir / "samples.pt",
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=None,  # lightweight checkpoint; two optimizers are used.
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_16_style_transfer_translation_cyclegan"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_16_style_transfer_translation_cyclegan.train"
        )
    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
