
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
from .model import LINE, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    steps_per_epoch: int = 200
    batch_size: int = 256
    negative_samples: int = 5
    learning_rate: float = 1e-2
    seed: int = 42
    device: str = "auto"
    run_name: str = "dev"

    embed_dim: int = 16
    order: int = 2


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Lesson 08 (GNN): LINE-style embeddings on Karate graph.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--order", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        negative_samples=args.negative_samples,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        run_name=args.run_name,
        embed_dim=args.embed_dim,
        order=args.order,
    )


def _make_negative_sampler(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # Degree-based negative sampling distribution: deg^0.75
    src = edge_index[0]
    deg = torch.bincount(src, minlength=int(num_nodes)).to(torch.float32)
    probs = torch.pow(deg, 0.75)
    probs = probs / probs.sum().clamp(min=1e-12)
    return probs


def run_training(cfg: TrainConfig) -> int:
    set_seed(cfg.seed)
    device_info = resolve_device(cfg.device)

    paths = build_run_paths(track="gnn", lesson="lesson_08_line_karate_embedding", run_name=cfg.run_name)
    logger = get_logger("gnn.karate_line", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    graph = load_karate(add_self_loops=False)
    edge_index = graph.edge_index.to(device_info.torch_device)
    num_nodes = int(graph.num_nodes)

    model = LINE(ModelConfig(num_nodes=num_nodes, embed_dim=cfg.embed_dim, order=cfg.order)).to(
        device_info.torch_device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    neg_probs = _make_negative_sampler(edge_index, num_nodes=num_nodes).to(device_info.torch_device)
    src_edges = edge_index[0]
    dst_edges = edge_index[1]
    num_edges = int(src_edges.numel())

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for step in range(int(cfg.steps_per_epoch)):
            edge_ids = torch.randint(0, num_edges, (int(cfg.batch_size),), device=device_info.torch_device)
            src = src_edges[edge_ids]
            dst = dst_edges[edge_ids]

            neg = torch.multinomial(
                neg_probs, num_samples=int(cfg.batch_size) * int(cfg.negative_samples), replacement=True
            ).view(int(cfg.batch_size), int(cfg.negative_samples))

            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(src=src, dst=dst, neg_dst=neg)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, int(cfg.steps_per_epoch))
        logger.info("Epoch %d/%d | loss %.4f", epoch, cfg.epochs, avg_loss)
        append_jsonl(metrics_path, {"epoch": epoch, "loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]})

    payload: dict[str, torch.Tensor] = {"node_embeddings": model.node_embeddings.weight.detach().cpu()}
    if model.context_embeddings is not None:
        payload["context_embeddings"] = model.context_embeddings.weight.detach().cpu()
    torch.save(payload, paths.run_dir / "embeddings.pt")

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=cfg.epochs,
        extra={"track": "gnn", "lesson": "lesson_08_line_karate_embedding"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.gnn.lesson_08_line_karate_embedding.train"
        )

    cfg = parse_args()
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

