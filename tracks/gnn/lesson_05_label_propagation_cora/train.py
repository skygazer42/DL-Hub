
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

from tracks.gnn.datasets.cora import load_cora

from .model import LabelPropagation, LabelPropagationConfig


@dataclass(frozen=True)
class TrainConfig:
    num_layers: int = 10
    alpha: float = 0.9
    clamp_labeled: bool = True

    seed: int = 42
    device: str = "auto"
    run_name: str = "dev"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Lesson 05 (GNN): Label Propagation baseline on Cora (pure PyTorch sparse)."
    )
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--clamp-labeled", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    args = parser.parse_args()

    return TrainConfig(
        num_layers=args.num_layers,
        alpha=args.alpha,
        clamp_labeled=args.clamp_labeled,
        seed=args.seed,
        device=args.device,
        run_name=args.run_name,
    )


def _accuracy(probs: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> float:
    pred = probs[idx].argmax(dim=1)
    return float((pred == labels[idx]).float().mean().item())


def run_training(cfg: TrainConfig) -> int:
    set_seed(cfg.seed)
    device_info = resolve_device(cfg.device)

    paths = build_run_paths(track="gnn", lesson="lesson_05_label_propagation_cora", run_name=cfg.run_name)
    logger = get_logger("gnn.cora_lp", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    data = load_cora()
    labels = data.labels.to(device_info.torch_device)
    adj_row = data.adj_row.to(device_info.torch_device)
    idx_train = data.idx_train.to(device_info.torch_device)
    idx_val = data.idx_val.to(device_info.torch_device)
    idx_test = data.idx_test.to(device_info.torch_device)

    model = LabelPropagation(
        LabelPropagationConfig(
            num_layers=cfg.num_layers, alpha=cfg.alpha, clamp_labeled=cfg.clamp_labeled
        )
    ).to(device_info.torch_device)

    metrics_path = paths.run_dir / "metrics.jsonl"
    # We log each propagation step for learning/diagnostics.
    probs = None
    for step in range(1, cfg.num_layers + 1):
        model.cfg = LabelPropagationConfig(
            num_layers=step, alpha=cfg.alpha, clamp_labeled=cfg.clamp_labeled
        )
        probs = model(adj_row=adj_row, labels=labels, idx_labeled=idx_train)

        train_acc = _accuracy(probs, labels, idx_train)
        val_acc = _accuracy(probs, labels, idx_val)
        test_acc = _accuracy(probs, labels, idx_test)

        logger.info(
            "Step %d/%d | train acc %.3f | val acc %.3f | test acc %.3f",
            step,
            cfg.num_layers,
            train_acc,
            val_acc,
            test_acc,
        )
        append_jsonl(
            metrics_path,
            {"step": step, "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc},
        )

    assert probs is not None
    torch.save({"probs": probs.cpu(), "labels": labels.cpu()}, paths.run_dir / "preds.pt")

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=None,
        epoch=cfg.num_layers,
        extra={"track": "gnn", "lesson": "lesson_05_label_propagation_cora"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.gnn.lesson_05_label_propagation_cora.train"
        )

    cfg = parse_args()
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

