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

from .data import load_karate
from .model import SDNE, ModelConfig, sdne_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    learning_rate: float = 1e-3
    lambda_smooth: float = 1.0
    seed: int = 42
    device: str = "auto"
    run_name: str = "dev"

    embed_dim: int = 16
    hidden_dim: int = 64
    dropout: float = 0.1


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Lesson 07 (GNN): SDNE-style embeddings on Karate graph."
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-smooth", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lambda_smooth=args.lambda_smooth,
        seed=args.seed,
        device=args.device,
        run_name=args.run_name,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )


def run_training(cfg: TrainConfig) -> int:
    set_seed(cfg.seed)
    device_info = resolve_device(cfg.device)

    paths = build_run_paths(
        track="gnn", lesson="lesson_07_sdne_karate_embedding", run_name=cfg.run_name
    )
    logger = get_logger("gnn.karate_sdne", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    graph = load_karate()
    adj = graph.adj.to(device_info.torch_device)
    edge_index = graph.edge_index.to(device_info.torch_device)

    model = SDNE(
        ModelConfig(
            num_nodes=int(graph.num_nodes),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )
    ).to(device_info.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        recon_logits, z = model(adj)
        loss, recon, smooth = sdne_loss(
            recon_logits=recon_logits,
            adj=adj,
            embeddings=z,
            edge_index=edge_index,
            lambda_smooth=cfg.lambda_smooth,
        )
        loss.backward()
        optimizer.step()

        logger.info(
            "Epoch %d/%d | loss %.4f (recon %.4f, smooth %.4f)",
            epoch,
            cfg.epochs,
            float(loss.item()),
            float(recon.item()),
            float(smooth.item()),
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "recon": float(recon.item()),
                "smooth": float(smooth.item()),
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    model.eval()
    with torch.no_grad():
        _, z = model(adj)
    torch.save({"embeddings": z.detach().cpu()}, paths.run_dir / "embeddings.pt")

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=cfg.epochs,
        extra={"track": "gnn", "lesson": "lesson_07_sdne_karate_embedding"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.gnn.lesson_07_sdne_karate_embedding.train"
        )

    cfg = parse_args()
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
