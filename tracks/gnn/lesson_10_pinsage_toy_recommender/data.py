from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DataConfig:
    num_users: int = 128
    num_items: int = 256
    true_dim: int = 8
    interactions_per_user: int = 10
    test_fraction: float = 0.2

    num_random_walks: int = 32  # per item, to estimate neighbors
    num_neighbors: int = 8

    seed: int = 0


@dataclass(frozen=True)
class ToyRecData:
    num_users: int
    num_items: int
    user_train_items: list[torch.Tensor]  # len U, each: (n_train,)
    user_test_items: list[torch.Tensor]  # len U, each: (n_test,)
    item_neighbors: torch.Tensor  # (I, K) padded with -1
    item_to_users: list[torch.Tensor]  # len I, users who interacted (train)


def _build_item_to_users(num_items: int, user_train_items: list[torch.Tensor]) -> list[list[int]]:
    buckets: list[list[int]] = [[] for _ in range(int(num_items))]
    for u, items in enumerate(user_train_items):
        for it in items.tolist():
            buckets[int(it)].append(int(u))
    return buckets


def _sample_item_neighbors(
    *,
    num_items: int,
    user_train_items: list[torch.Tensor],
    item_to_users: list[torch.Tensor],
    num_random_walks: int,
    num_neighbors: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """Estimate item-item neighbors by short random walks: item -> user -> item."""

    neighbors = torch.full((int(num_items), int(num_neighbors)), fill_value=-1, dtype=torch.long)

    for item_id in range(int(num_items)):
        users = item_to_users[item_id]
        if int(users.numel()) == 0:
            continue

        counts: dict[int, int] = {}
        for _ in range(int(num_random_walks)):
            u = int(users[torch.randint(0, int(users.numel()), (1,), generator=gen)].item())
            items = user_train_items[u]
            if int(items.numel()) == 0:
                continue
            j = int(items[torch.randint(0, int(items.numel()), (1,), generator=gen)].item())
            if j == item_id:
                continue
            counts[j] = counts.get(j, 0) + 1

        if not counts:
            continue

        # Top-K by frequency
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: int(num_neighbors)]
        neighbors[item_id, : len(top)] = torch.tensor([k for k, _ in top], dtype=torch.long)

    return neighbors


def build_toy_recommender_data(cfg: DataConfig) -> ToyRecData:
    """Create a tiny implicit-feedback dataset from latent factors."""

    if not (0.0 < float(cfg.test_fraction) < 1.0):
        raise ValueError("test_fraction must be in (0, 1)")

    gen = torch.Generator().manual_seed(int(cfg.seed))
    num_users, num_items = int(cfg.num_users), int(cfg.num_items)
    true_dim = int(cfg.true_dim)

    # Ground-truth latent factors that generate interactions.
    u_true = torch.randn((num_users, true_dim), generator=gen)
    v_true = torch.randn((num_items, true_dim), generator=gen)

    user_train_items: list[torch.Tensor] = []
    user_test_items: list[torch.Tensor] = []

    n_total = int(cfg.interactions_per_user)
    n_test = max(1, int(round(n_total * float(cfg.test_fraction))))
    n_train = n_total - n_test
    if n_train < 1:
        raise ValueError("interactions_per_user too small for the chosen test_fraction")

    noise = 0.10
    for u in range(num_users):
        scores = (u_true[u].unsqueeze(0) * v_true).sum(dim=1) + noise * torch.randn(
            (num_items,), generator=gen
        )
        top_items = torch.topk(scores, k=n_total, largest=True).indices.to(torch.long)
        perm = torch.randperm(n_total, generator=gen)
        top_items = top_items[perm]

        train = top_items[:n_train].clone()
        test = top_items[n_train:].clone()
        user_train_items.append(train)
        user_test_items.append(test)

    # Build item->users adjacency (train only).
    item_to_users_lists = _build_item_to_users(num_items, user_train_items)
    item_to_users = [
        torch.tensor(lst, dtype=torch.long) if lst else torch.empty((0,), dtype=torch.long)
        for lst in item_to_users_lists
    ]

    item_neighbors = _sample_item_neighbors(
        num_items=num_items,
        user_train_items=user_train_items,
        item_to_users=item_to_users,
        num_random_walks=int(cfg.num_random_walks),
        num_neighbors=int(cfg.num_neighbors),
        gen=gen,
    )

    return ToyRecData(
        num_users=num_users,
        num_items=num_items,
        user_train_items=user_train_items,
        user_test_items=user_test_items,
        item_neighbors=item_neighbors,
        item_to_users=item_to_users,
    )


__all__ = ["DataConfig", "ToyRecData", "build_toy_recommender_data"]
