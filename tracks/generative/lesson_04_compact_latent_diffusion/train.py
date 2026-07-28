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
from .model import LatentDiffusionModel, ModelConfig, diffusion_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    learning_rate: float = 1e-3
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
    parser = argparse.ArgumentParser(
        description="Lesson 04 (Generative): compact latent diffusion with a compact autoencoder."
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)

    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--num-diffusion-steps", type=int, default=8)
    parser.add_argument("--recon-weight", type=float, default=0.25)

    args = parser.parse_args()

    train_cfg = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        run_name=args.run_name,
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
        latent_channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
        num_diffusion_steps=args.num_diffusion_steps,
        recon_weight=args.recon_weight,
    )
    return train_cfg, data_cfg, model_cfg


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig, model_cfg: ModelConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(
        track="generative", lesson="lesson_04_compact_latent_diffusion", run_name=train_cfg.run_name
    )
    logger = get_logger("generative.compact_latent_diffusion", log_file=paths.logs_dir / "train.log")
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
    model = LatentDiffusionModel(model_cfg).to(device_info.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_idx, images in enumerate(train_loader):
            if train_cfg.max_train_batches is not None and batch_idx >= train_cfg.max_train_batches:
                break

            images = images.to(device_info.torch_device)
            optimizer.zero_grad(set_to_none=True)

            latents = model.encode(images)
            timesteps = torch.randint(
                0,
                model_cfg.num_diffusion_steps,
                (images.size(0),),
                device=device_info.torch_device,
                dtype=torch.long,
            )
            noisy_latents, noise = model.add_noise(latents, timesteps)
            noise_pred = model.predict_noise(noisy_latents, timesteps)
            recon_images = model.decode(latents)

            loss = diffusion_loss(
                noise_pred=noise_pred,
                noise=noise,
                recon_images=recon_images,
                target_images=images,
                recon_weight=model_cfg.recon_weight,
            )
            loss.backward()
            optimizer.step()

            batch_size = int(images.size(0))
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(1, train_count)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_idx, images in enumerate(val_loader):
                if train_cfg.max_eval_batches is not None and batch_idx >= train_cfg.max_eval_batches:
                    break

                images = images.to(device_info.torch_device)
                latents = model.encode(images)
                timesteps = torch.randint(
                    0,
                    model_cfg.num_diffusion_steps,
                    (images.size(0),),
                    device=device_info.torch_device,
                    dtype=torch.long,
                )
                noisy_latents, noise = model.add_noise(latents, timesteps)
                noise_pred = model.predict_noise(noisy_latents, timesteps)
                recon_images = model.decode(latents)
                loss = diffusion_loss(
                    noise_pred=noise_pred,
                    noise=noise,
                    recon_images=recon_images,
                    target_images=images,
                    recon_weight=model_cfg.recon_weight,
                )
                batch_size = int(images.size(0))
                val_loss_sum += float(loss.item()) * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / max(1, val_count)
        logger.info(
            "Epoch %d/%d | train loss %.4f | val loss %.4f",
            epoch,
            train_cfg.epochs,
            train_loss,
            val_loss,
        )
        append_jsonl(
            metrics_path,
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss},
        )

        with torch.no_grad():
            samples = model.sample(batch_size=16, device=device_info.torch_device).cpu()
        torch.save({"samples": samples}, paths.run_dir / "samples.pt")
        _maybe_save_image_grid(samples, paths.run_dir / "samples.png")

    model.eval()
    with torch.no_grad():
        images = next(iter(val_loader)).to(device_info.torch_device)
        latents = model.encode(images)
        recons = model.decode(latents).cpu()
        images = images.cpu()
    torch.save({"inputs": images, "reconstructions": recons}, paths.run_dir / "recons.pt")
    _maybe_save_image_grid(torch.cat([images[:16], recons[:16]], dim=0), paths.run_dir / "recons.png")

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=train_cfg.epochs,
        extra={"track": "generative", "lesson": "lesson_04_compact_latent_diffusion"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.generative.lesson_04_compact_latent_diffusion.train"
        )
    train_cfg, data_cfg, model_cfg = parse_args()
    return run_training(train_cfg, data_cfg, model_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
