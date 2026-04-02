import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from dlhub.config import append_jsonl, dataclass_to_dict, write_json
from dlhub.device import resolve_device
from dlhub.logging import get_logger
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed
from dlhub.vision.style_transfer_zoo import build_local_model

from .data import DataConfig, make_batch


@dataclass(frozen=True)
class RunConfig:
    arch: str = "dlst:gatys_tiny"
    steps: int = 8
    lr: float = 0.03
    content_weight: float = 1.0
    style_weight: float = 5.0
    tv_weight: float = 1e-3

    seed: int = 42
    device: str = "auto"
    run_name: str = "dev"

    width_mult: float = 1.0
    dropout: float = 0.0


def _maybe_save_image(image: torch.Tensor, path: str | Path) -> None:
    try:
        from torchvision.utils import save_image
    except Exception:
        return
    save_image(image, path)


def parse_args() -> tuple[RunConfig, DataConfig]:
    parser = argparse.ArgumentParser(description="Lesson 15 (Vision): Gatys-style neural style transfer (toy-first).")

    parser.add_argument("--arch", type=str, default="dlst:gatys_tiny")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=5.0)
    parser.add_argument("--tv-weight", type=float, default=1e-3)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--data-seed", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    args = parser.parse_args()

    run_cfg = RunConfig(
        arch=args.arch,
        steps=args.steps,
        lr=args.lr,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        seed=args.seed,
        device=args.device,
        run_name=args.run_name,
        width_mult=args.width_mult,
        dropout=args.dropout,
    )
    data_cfg = DataConfig(
        batch_size=args.batch_size,
        image_size=args.image_size,
        in_channels=args.in_channels,
        seed=args.data_seed,
    )
    return run_cfg, data_cfg


def run_style_transfer(run_cfg: RunConfig, data_cfg: DataConfig) -> int:
    set_seed(run_cfg.seed)
    device_info = resolve_device(run_cfg.device)

    paths = build_run_paths(
        track="vision", lesson="lesson_15_neural_style_transfer_gatys", run_name=run_cfg.run_name
    )
    logger = get_logger("vision.style_transfer_gatys", log_file=paths.logs_dir / "run.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Arch: %s", run_cfg.arch)
    logger.info("Outputs: %s", paths.run_dir)

    write_json(
        paths.run_dir / "config.json",
        {
            "run": dataclass_to_dict(run_cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    content, style = make_batch(data_cfg)
    content = content.to(device_info.torch_device)
    style = style.to(device_info.torch_device)

    model = build_local_model(
        run_cfg.arch,
        in_channels=int(data_cfg.in_channels),
        image_size=int(data_cfg.image_size),
        width_mult=float(run_cfg.width_mult),
        dropout=float(run_cfg.dropout),
        steps=int(run_cfg.steps),
        lr=float(run_cfg.lr),
        content_weight=float(run_cfg.content_weight),
        style_weight=float(run_cfg.style_weight),
        tv_weight=float(run_cfg.tv_weight),
    ).to(device_info.torch_device)

    model.eval()
    out = model(content, style)
    stylized = out["stylized"].detach().cpu()
    loss = out.get("loss", torch.tensor(0.0)).detach().cpu()

    metrics_path = paths.run_dir / "metrics.jsonl"
    append_jsonl(metrics_path, {"loss": float(loss.item())})
    logger.info("Final loss: %.6f", float(loss.item()))

    torch.save({"content": content.detach().cpu(), "style": style.detach().cpu(), "stylized": stylized}, paths.run_dir / "stylized.pt")
    _maybe_save_image(stylized[:1], paths.run_dir / "stylized.png")
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.vision.lesson_15_neural_style_transfer_gatys.train"
        )
    run_cfg, data_cfg = parse_args()
    return run_style_transfer(run_cfg, data_cfg)


if __name__ == "__main__":
    raise SystemExit(main())

