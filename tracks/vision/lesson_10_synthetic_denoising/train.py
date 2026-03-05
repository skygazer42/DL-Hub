from __future__ import annotations

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
from dlhub.training.loop import fit_regression

from .data import DataConfig, get_dataloaders
from .model import DenoiserAdapter, ModelConfig, build_model, list_supported_arches


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    run_name: str = "dev"

    arch: str = "dncnn:dncnn_9"
    in_channels: int = 1
    sigma: float = 0.1  # used by bm3d


def parse_args() -> tuple[TrainConfig, DataConfig]:
    parser = argparse.ArgumentParser(description="Lesson 10 (Vision): Synthetic image denoising.")

    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument(
        "--noise-type",
        type=str,
        default="gaussian",
        help="gaussian | gaussian_var | gaussian_impulse | poisson | impulse | shot_read | speckle | speckle_read | stripe",
    )
    parser.add_argument("--noise-std", type=float, default=0.1, help="Gaussian noise std (used when noise-type=gaussian)")
    parser.add_argument("--noise-std-min", type=float, default=0.05, help="Min Gaussian std (used when noise-type=gaussian_var)")
    parser.add_argument("--noise-std-max", type=float, default=0.2, help="Max Gaussian std (used when noise-type=gaussian_var)")
    parser.add_argument("--poisson-peak", type=float, default=30.0, help="Poisson peak photons (used when noise-type=poisson)")
    parser.add_argument("--impulse-prob", type=float, default=0.03, help="Salt & pepper prob (used when noise-type=impulse)")
    parser.add_argument("--shot-noise", type=float, default=0.2, help="Shot noise factor (used when noise-type=shot_read)")
    parser.add_argument("--read-noise", type=float, default=0.02, help="Read noise std (used when noise-type=shot_read)")
    parser.add_argument("--speckle-std", type=float, default=0.15, help="Speckle std (used when noise-type=speckle/speckle_read)")
    parser.add_argument("--stripe-amplitude", type=float, default=0.12, help="Stripe amplitude (used when noise-type=stripe)")
    parser.add_argument("--stripe-period", type=int, default=8, help="Stripe period in pixels (used when noise-type=stripe)")
    parser.add_argument("--stripe-direction", type=str, default="vertical", help="vertical | horizontal | random (used when noise-type=stripe)")
    parser.add_argument("--min-square", type=int, default=8)
    parser.add_argument("--max-square", type=int, default=24)
    parser.add_argument("--train-mode", type=str, default="supervised", help="supervised | noise2noise | blindspot")
    parser.add_argument("--blindspot-prob", type=float, default=0.1, help="Masking probability for blindspot mode (0,1).")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--arch",
        type=str,
        default="dncnn:dncnn_9",
        help="Examples: dncnn:dncnn_17 | restormer:restormer_tiny | noise2noise_unet:n2n_unet_tiny | bm3d:bm3d_fast | cbdnet:cbdnet_tiny",
    )
    parser.add_argument("--list-arch", action="store_true", help="Print supported architectures and exit.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise sigma for BM3D baseline (in [0,1] scale).")

    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--run-name", type=str, default="dev")
    args = parser.parse_args()

    if args.list_arch:
        print("\n".join(list_supported_arches()))
        raise SystemExit(0)

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
        sigma=args.sigma,
    )
    data_cfg = DataConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.data_seed,
        num_workers=args.num_workers,
        in_channels=args.in_channels,
        noise_type=args.noise_type,
        noise_std=args.noise_std,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        poisson_peak=args.poisson_peak,
        impulse_prob=args.impulse_prob,
        shot_noise=args.shot_noise,
        read_noise=args.read_noise,
        speckle_std=args.speckle_std,
        stripe_amplitude=args.stripe_amplitude,
        stripe_period=args.stripe_period,
        stripe_direction=args.stripe_direction,
        min_square=args.min_square,
        max_square=args.max_square,
        train_mode=args.train_mode,
        blindspot_prob=args.blindspot_prob,
    )
    return train_cfg, data_cfg


def _psnr(pred: torch.Tensor, target: torch.Tensor, *, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target).pow(2), dim=(1, 2, 3))
    psnr = 10.0 * torch.log10((max_val * max_val) / mse.clamp_min(1e-10))
    return psnr


def evaluate_denoiser(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Return (mse, psnr) averaged over the loader."""

    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    total = 0

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = model(noisy)
            mse = torch.mean((pred - clean).pow(2), dim=(1, 2, 3))
            psnr = _psnr(pred, clean)
            bs = int(noisy.size(0))
            total += bs
            total_mse += float(mse.mean().item()) * bs
            total_psnr += float(psnr.mean().item()) * bs

    if total == 0:
        return 0.0, 0.0
    return total_mse / total, total_psnr / total


def run_training(train_cfg: TrainConfig, data_cfg: DataConfig) -> int:
    set_seed(train_cfg.seed)
    device_info = resolve_device(train_cfg.device)

    paths = build_run_paths(track="vision", lesson="lesson_10_synthetic_denoising", run_name=train_cfg.run_name)
    logger = get_logger("vision.denoising", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Arch: %s", train_cfg.arch)
    logger.info("Train mode: %s", data_cfg.train_mode)
    logger.info("Outputs: %s", paths.run_dir)

    train_loader, val_loader = get_dataloaders(data_cfg)
    model_cfg = ModelConfig(arch=train_cfg.arch, variant="", in_channels=train_cfg.in_channels, sigma=train_cfg.sigma)
    model = DenoiserAdapter(build_model(model_cfg)).to(device_info.torch_device)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(train_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    # BM3D is a non-learnable baseline; just evaluate and exit.
    if str(train_cfg.arch).lower().strip().startswith("bm3d:"):
        mse, psnr = evaluate_denoiser(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
        )
        logger.info("BM3D eval | mse %.6f | psnr %.2f dB", mse, psnr)
        write_json(paths.run_dir / "metrics.json", {"eval_mse": mse, "eval_psnr": psnr})
        return 0

    if str(data_cfg.train_mode).lower().strip() == "blindspot":
        from .losses import MaskedMSELoss

        criterion = MaskedMSELoss()
    else:
        criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.learning_rate))

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, int(train_cfg.epochs) + 1):
        train_stats = fit_regression(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device_info.torch_device,
            max_batches=train_cfg.max_train_batches,
        )
        eval_mse, eval_psnr = evaluate_denoiser(
            model=model,
            loader=val_loader,
            device=device_info.torch_device,
            max_batches=train_cfg.max_eval_batches,
        )

        logger.info(
            "Epoch %d/%d | train mse %.6f | eval mse %.6f psnr %.2f dB",
            epoch,
            train_cfg.epochs,
            train_stats.loss,
            eval_mse,
            eval_psnr,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_mse": train_stats.loss,
                "eval_mse": eval_mse,
                "eval_psnr": eval_psnr,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(train_cfg.epochs),
        extra={"track": "vision", "lesson": "lesson_10_synthetic_denoising"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_10_synthetic_denoising.train"
        )

    train_cfg, data_cfg = parse_args()
    return run_training(train_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
