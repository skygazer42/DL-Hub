
import argparse
import sys
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from dlhub.checkpoint import save_checkpoint
from dlhub.config import append_jsonl, dataclass_to_dict, write_json
from dlhub.device import resolve_device
from dlhub.logging import get_logger
from dlhub.paths import build_run_paths
from dlhub.seed import set_seed

from .data import DataConfig, build_training_pairs
from .model import MetaPath2Vec, ModelConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    batch_size: int = 512
    negative_samples: int = 5
    learning_rate: float = 2.5e-2
    seed: int = 42
    device: str = "auto"
    run_name: str = "dev"

    embed_dim: int = 64
    sparse: bool = True

    # Data params (kept here for a single-file CLI experience)
    num_authors: int = 24
    num_papers: int = 48
    num_venues: int = 6
    authors_per_paper: int = 2
    papers_per_author: int = 4
    metapath: str = "A2P,P2A"
    start_type: str = "author"
    num_walks: int = 200
    walk_length: int = 30
    window_size: int = 5
    care_type: int = 0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Lesson 09 (GNN): metapath2vec-style embeddings on a toy heterogeneous graph."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2.5e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--dense", action="store_true", help="Use dense Adam instead of SparseAdam")

    # Graph/walk config
    parser.add_argument("--num-authors", type=int, default=24)
    parser.add_argument("--num-papers", type=int, default=48)
    parser.add_argument("--num-venues", type=int, default=6)
    parser.add_argument("--authors-per-paper", type=int, default=2)
    parser.add_argument("--papers-per-author", type=int, default=4)
    parser.add_argument("--metapath", type=str, default="A2P,P2A")
    parser.add_argument("--start-type", type=str, default="author")
    parser.add_argument("--num-walks", type=int, default=200)
    parser.add_argument("--walk-length", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument(
        "--care-type",
        type=int,
        default=0,
        choices=[0, 1],
        help="0: global negatives; 1: same-type negatives (metapath2vec++ style)",
    )
    args = parser.parse_args()

    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        negative_samples=args.negative_samples,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        run_name=args.run_name,
        embed_dim=args.embed_dim,
        sparse=not args.dense,
        num_authors=args.num_authors,
        num_papers=args.num_papers,
        num_venues=args.num_venues,
        authors_per_paper=args.authors_per_paper,
        papers_per_author=args.papers_per_author,
        metapath=args.metapath,
        start_type=args.start_type,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        window_size=args.window_size,
        care_type=args.care_type,
    )


def _to_data_cfg(cfg: TrainConfig) -> DataConfig:
    return DataConfig(
        num_authors=cfg.num_authors,
        num_papers=cfg.num_papers,
        num_venues=cfg.num_venues,
        authors_per_paper=cfg.authors_per_paper,
        papers_per_author=cfg.papers_per_author,
        metapath=cfg.metapath,
        start_type=cfg.start_type,
        num_walks=cfg.num_walks,
        walk_length=cfg.walk_length,
        window_size=cfg.window_size,
        care_type=cfg.care_type,
        seed=cfg.seed,
    )


def run_training(cfg: TrainConfig) -> int:
    set_seed(cfg.seed)
    device_info = resolve_device(cfg.device)

    paths = build_run_paths(
        track="gnn", lesson="lesson_09_metapath2vec_toy_hetero_embedding", run_name=cfg.run_name
    )
    logger = get_logger("gnn.metapath2vec_toy", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device_info.torch_device)
    logger.info("Outputs: %s", paths.run_dir)

    data_cfg = _to_data_cfg(cfg)
    graph, pairs, neg_sampler = build_training_pairs(data_cfg)

    loader = DataLoader(
        pairs,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    model = MetaPath2Vec(ModelConfig(num_nodes=graph.num_nodes, embed_dim=cfg.embed_dim, sparse=cfg.sparse)).to(
        device_info.torch_device
    )
    optimizer: torch.optim.Optimizer
    if cfg.sparse:
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=cfg.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
            "graph": {"type_names": list(graph.type_names), "num_nodes": graph.num_nodes},
        },
    )

    metrics_path = paths.run_dir / "metrics.jsonl"
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0

        for center, context in loader:
            neg = neg_sampler.sample(context, k=int(cfg.negative_samples))
            center = center.to(device_info.torch_device)
            context = context.to(device_info.torch_device)
            neg = neg.to(device_info.torch_device)

            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(center=center, context=context, neg_context=neg)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_steps += 1

        avg_loss = total_loss / max(1, total_steps)
        logger.info("Epoch %d/%d | loss %.4f", epoch, cfg.epochs, avg_loss)
        append_jsonl(metrics_path, {"epoch": epoch, "loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]})

    torch.save(
        {
            "u_embeddings": model.u_embeddings.weight.detach().cpu(),
            "v_embeddings": model.v_embeddings.weight.detach().cpu(),
            "node_types": graph.node_types.detach().cpu(),
            "type_names": list(graph.type_names),
            "node_names": graph.node_names,
        },
        paths.run_dir / "embeddings.pt",
    )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(cfg.epochs),
        extra={"track": "gnn", "lesson": "lesson_09_metapath2vec_toy_hetero_embedding"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.gnn.lesson_09_metapath2vec_toy_hetero_embedding.train"
        )

    cfg = parse_args()
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
