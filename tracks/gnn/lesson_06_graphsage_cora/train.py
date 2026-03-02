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

from tracks.gnn.datasets.cora import load_cora

from .model import GraphSAGE, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    seed: int = 42
    device: str = "auto"
    run_name: str = "dev"

    hidden_features: int = 64
    dropout: float = 0.5


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Lesson 06 (GNN): GraphSAGE on Cora (full-batch).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")
    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_features=args.hidden_features,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
        run_name=args.run_name,
    )


def run_training(cfg: TrainConfig) -> int:
    set_seed(cfg.seed)
    device_info = resolve_device(cfg.device)

    paths = build_run_paths(track="gnn", lesson="lesson_06_graphsage_cora", run_name=cfg.run_name)
    logger = get_logger("gnn.cora_graphsage", log_file=paths.logs_dir / "train.log")
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
    features = data.features.to(device_info.torch_device)
    labels = data.labels.to(device_info.torch_device)
    adj_row = data.adj_row.to(device_info.torch_device)
    idx_train = data.idx_train.to(device_info.torch_device)
    idx_val = data.idx_val.to(device_info.torch_device)
    idx_test = data.idx_test.to(device_info.torch_device)

    model = GraphSAGE(
        ModelConfig(
            in_features=int(features.shape[1]),
            hidden_features=cfg.hidden_features,
            num_classes=int(labels.max().item()) + 1,
            dropout=cfg.dropout,
        )
    ).to(device_info.torch_device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, adj_row)
        loss = criterion(logits[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(features, adj_row)
            train_acc = float(
                (logits[idx_train].argmax(dim=1) == labels[idx_train]).float().mean().item()
            )
            val_loss = float(criterion(logits[idx_val], labels[idx_val]).item())
            val_acc = float((logits[idx_val].argmax(dim=1) == labels[idx_val]).float().mean().item())
            test_acc = float((logits[idx_test].argmax(dim=1) == labels[idx_test]).float().mean().item())

        logger.info(
            "Epoch %d/%d | train loss %.4f acc %.3f | val loss %.4f acc %.3f | test acc %.3f",
            epoch,
            cfg.epochs,
            float(loss.item()),
            train_acc,
            val_loss,
            val_acc,
            test_acc,
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=cfg.epochs,
        extra={"track": "gnn", "lesson": "lesson_06_graphsage_cora"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.gnn.lesson_06_graphsage_cora.train"
        )

    cfg = parse_args()
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

