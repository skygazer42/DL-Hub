
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

from .data import DataConfig, ToyRecData, build_toy_recommender_data
from .model import ModelConfig, PinSAGEItemEncoder


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    steps_per_epoch: int = 200
    batch_size: int = 256
    negative_samples: int = 10
    learning_rate: float = 2e-3
    seed: int = 42
    device: str = "auto"
    run_name: str = "dev"

    embed_dim: int = 64
    num_neighbors: int = 8
    normalize: bool = True

    # Data params
    num_users: int = 128
    num_items: int = 256
    interactions_per_user: int = 10
    test_fraction: float = 0.2
    num_random_walks: int = 32

    eval_k: int = 20


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Lesson 10 (GNN): PinSAGE-style toy recommender.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negative-samples", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="dev")

    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-neighbors", type=int, default=8)
    parser.add_argument("--no-normalize", action="store_true")

    parser.add_argument("--num-users", type=int, default=128)
    parser.add_argument("--num-items", type=int, default=256)
    parser.add_argument("--interactions-per-user", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--num-random-walks", type=int, default=32)
    parser.add_argument("--eval-k", type=int, default=20)
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
        num_neighbors=args.num_neighbors,
        normalize=not args.no_normalize,
        num_users=args.num_users,
        num_items=args.num_items,
        interactions_per_user=args.interactions_per_user,
        test_fraction=args.test_fraction,
        num_random_walks=args.num_random_walks,
        eval_k=args.eval_k,
    )


def _to_data_cfg(cfg: TrainConfig) -> DataConfig:
    return DataConfig(
        num_users=cfg.num_users,
        num_items=cfg.num_items,
        interactions_per_user=cfg.interactions_per_user,
        test_fraction=cfg.test_fraction,
        num_random_walks=cfg.num_random_walks,
        num_neighbors=cfg.num_neighbors,
        seed=cfg.seed,
    )


def _sample_pos_items(
    *,
    item_ids: torch.Tensor,
    item_neighbors: torch.Tensor,
    gen: torch.Generator,
) -> torch.Tensor:
    """Pick one positive neighbor per item from the precomputed neighbor table."""

    item_ids = item_ids.to(torch.long)
    neigh = item_neighbors[item_ids]  # (B, K) with -1
    b, k = neigh.shape

    # Fallback: if a row has no neighbors, use itself.
    has_any = (neigh >= 0).any(dim=1)
    pos = torch.empty((b,), dtype=torch.long)

    for i in range(int(b)):
        if bool(has_any[i].item()):
            valid = neigh[i][neigh[i] >= 0]
            pos[i] = valid[torch.randint(0, int(valid.numel()), (1,), generator=gen)].item()
        else:
            pos[i] = item_ids[i].item()
    return pos


@torch.no_grad()
def _eval_recall_at_k(
    *,
    data: ToyRecData,
    model: PinSAGEItemEncoder,
    device: torch.device,
    k: int,
) -> float:
    model.eval()
    item_ids = torch.arange(data.num_items, dtype=torch.long, device=device)
    item_neighbors = data.item_neighbors.to(device)
    item_repr = model.encode(item_ids=item_ids, neighbors=item_neighbors[item_ids])  # (I, D)

    hits = 0
    total = 0
    for u in range(data.num_users):
        train_items = data.user_train_items[u].to(device)
        test_items = data.user_test_items[u].to(device)
        if int(test_items.numel()) == 0:
            continue

        user_vec = item_repr[train_items].mean(dim=0, keepdim=True)  # (1, D)
        scores = (user_vec * item_repr).sum(dim=1)  # (I,)
        scores[train_items] = -1e9
        topk = torch.topk(scores, k=min(int(k), data.num_items), largest=True).indices

        rec = set(topk.tolist())
        for it in test_items.tolist():
            total += 1
            if int(it) in rec:
                hits += 1
    return float(hits) / max(1, total)


def run_training(cfg: TrainConfig) -> int:
    set_seed(cfg.seed)
    device_info = resolve_device(cfg.device)
    device = device_info.torch_device

    paths = build_run_paths(track="gnn", lesson="lesson_10_pinsage_toy_recommender", run_name=cfg.run_name)
    logger = get_logger("gnn.pinsage_toy", log_file=paths.logs_dir / "train.log")
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s (%s)", device_info.name, device)
    logger.info("Outputs: %s", paths.run_dir)

    data_cfg = _to_data_cfg(cfg)
    data = build_toy_recommender_data(data_cfg)

    model = PinSAGEItemEncoder(
        ModelConfig(
            num_items=data.num_items, embed_dim=cfg.embed_dim, num_neighbors=cfg.num_neighbors, normalize=cfg.normalize
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    write_json(
        paths.run_dir / "config.json",
        {
            "train": dataclass_to_dict(cfg),
            "data": dataclass_to_dict(data_cfg),
            "versions": {"python": sys.version, "torch": torch.__version__},
        },
    )

    gen = torch.Generator().manual_seed(int(cfg.seed) + 123)
    item_neighbors = data.item_neighbors.to(device)
    metrics_path = paths.run_dir / "metrics.jsonl"

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        total_loss = 0.0

        for _ in range(int(cfg.steps_per_epoch)):
            item_ids = torch.randint(low=0, high=data.num_items, size=(int(cfg.batch_size),), generator=gen, device=device)
            pos_ids = _sample_pos_items(item_ids=item_ids.cpu(), item_neighbors=data.item_neighbors, gen=gen).to(device)
            neg_ids = torch.randint(
                low=0,
                high=data.num_items,
                size=(int(cfg.batch_size), int(cfg.negative_samples)),
                generator=gen,
                device=device,
            )

            center = model.encode(item_ids=item_ids, neighbors=item_neighbors[item_ids])
            pos = model.encode(item_ids=pos_ids, neighbors=item_neighbors[pos_ids])

            b, k = neg_ids.shape
            neg_flat = neg_ids.reshape(-1)
            neg_repr = model.encode(item_ids=neg_flat, neighbors=item_neighbors[neg_flat]).view(b, k, -1)

            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(center=center, pos=pos, neg=neg_repr)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, int(cfg.steps_per_epoch))
        recall_at_k = _eval_recall_at_k(data=data, model=model, device=device, k=int(cfg.eval_k))
        logger.info("Epoch %d/%d | loss %.4f | recall@%d %.3f", epoch, cfg.epochs, avg_loss, cfg.eval_k, recall_at_k)
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "loss": avg_loss,
                "recall_at_k": recall_at_k,
                "k": int(cfg.eval_k),
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    with torch.no_grad():
        item_ids = torch.arange(data.num_items, dtype=torch.long, device=device)
        item_repr = model.encode(item_ids=item_ids, neighbors=item_neighbors[item_ids]).detach().cpu()

    torch.save(
        {"item_repr": item_repr, "neighbors": data.item_neighbors.cpu()},
        paths.run_dir / "embeddings.pt",
    )

    ckpt_path = save_checkpoint(
        paths.checkpoints_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        epoch=int(cfg.epochs),
        extra={"track": "gnn", "lesson": "lesson_10_pinsage_toy_recommender"},
    )
    logger.info("Saved checkpoint to %s", ckpt_path)
    return 0


def main() -> int:
    if __package__ is None:
        raise RuntimeError(
            "Please run this lesson from the repo root as a module:\n"
            "  python -m tracks.gnn.lesson_10_pinsage_toy_recommender.train"
        )
    cfg = parse_args()
    return run_training(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

