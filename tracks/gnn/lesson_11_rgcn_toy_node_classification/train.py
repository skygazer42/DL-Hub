
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

from .data import DataConfig, load_toy_rel_graph
from .model import ModelConfig, RGCN


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 150
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "auto"
    run_name: str = "dev"

    hidden_features: int = 32
    dropout: float = 0.1
    num_bases: int = -1

    # Data params
    num_nodes: int = 180
    num_rels: int = 4
    num_classes: int = 3
    feature_dim: int = 16
    edges_per_node: int = 4


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Lesson 11 (GNN): R-GCN on a toy relational graph.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--hidden-features", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-bases", type=int, default=-1)

    parser.add_argument("--num-nodes", type=int, default=180)
    parser.add_argument("--num-rels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--edges-per-node", type=int, default=4)
    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        run_name=args.run_name,
        hidden_features=args.hidden_features,
        dropout=args.dropout,
        num_bases=args.num_bases,
        num_nodes=args.num_nodes,
        num_rels=args.num_rels,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        edges_per_node=args.edges_per_node,
    )


def _to_data_cfg(cfg: TrainConfig) -> DataConfig:
    return DataConfig(
        num_nodes=cfg.num_nodes,
        num_rels=cfg.num_rels,
        num_classes=cfg.num_classes,
        feature_dim=cfg.feature_dim,
        edges_per_node=cfg.edges_per_node,
        seed=cfg.seed,
    )


def run_training(cfg: TrainConfig) -> int:
    set_seed(cfg.seed)
    device_info = resolve_device(cfg.device)

    paths = build_run_paths(track="gnn", lesson="lesson_11_rgcn_toy_node_classification", run_name=cfg.run_name)
    logger = get_logger("gnn.rgcn_toy", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    data_cfg = _to_data_cfg(cfg)
    data = load_toy_rel_graph(data_cfg)

    x = data.features.to(device_info.torch_device)
    y = data.labels.to(device_info.torch_device)
    edge_index = data.edge_index.to(device_info.torch_device)
    edge_type = data.edge_type.to(device_info.torch_device)
    edge_norm = data.edge_norm.to(device_info.torch_device)

    idx_train = data.idx_train.to(device_info.torch_device)
    idx_val = data.idx_val.to(device_info.torch_device)
    idx_test = data.idx_test.to(device_info.torch_device)

    model = RGCN(
        ModelConfig(
            in_features=int(x.shape[1]),
            hidden_features=int(cfg.hidden_features),
            num_classes=int(data.num_classes),
            num_rels=int(data.num_rels),
            num_bases=int(cfg.num_bases),
            dropout=float(cfg.dropout),
        )
    ).to(device_info.torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edge_index=edge_index, edge_type=edge_type, edge_norm=edge_norm)
        loss = criterion(logits[idx_train], y[idx_train])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index=edge_index, edge_type=edge_type, edge_norm=edge_norm)
            train_acc = float((logits[idx_train].argmax(dim=1) == y[idx_train]).float().mean().item())
            val_loss = float(criterion(logits[idx_val], y[idx_val]).item())
            val_acc = float((logits[idx_val].argmax(dim=1) == y[idx_val]).float().mean().item())
            test_acc = float((logits[idx_test].argmax(dim=1) == y[idx_test]).float().mean().item())

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
        epoch=int(cfg.epochs),
        extra={"track": "gnn", "lesson": "lesson_11_rgcn_toy_node_classification"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.gnn.lesson_11_rgcn_toy_node_classification.train"
        )
    cfg = parse_args()
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

