from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    prompt_token_id: int = 1
    rank_hi_token_id: int = 2
    rank_mid_token_id: int = 3
    rank_lo_token_id: int = 4
    separator_token_id: int = 5
    eos_id: int = 6
    base_vocab_size: int = 64
    ignore_index: int = -100

    @property
    def content_start_id(self) -> int:
        return 7

    @property
    def size(self) -> int:
        return int(self.content_start_id + self.base_vocab_size)

    def to_dict(self) -> dict[str, int]:
        return {
            "pad_id": int(self.pad_id),
            "prompt_token_id": int(self.prompt_token_id),
            "rank_hi_token_id": int(self.rank_hi_token_id),
            "rank_mid_token_id": int(self.rank_mid_token_id),
            "rank_lo_token_id": int(self.rank_lo_token_id),
            "separator_token_id": int(self.separator_token_id),
            "eos_id": int(self.eos_id),
            "base_vocab_size": int(self.base_vocab_size),
            "ignore_index": int(self.ignore_index),
        }


@dataclass(frozen=True)
class DataConfig:
    num_prompts: int = 1024
    group_size: int = 4
    batch_size: int = 32
    seq_length: int = 24
    base_vocab_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _next_content_token(token_id: int, vocab: Vocab) -> int:
    base = int(token_id - vocab.content_start_id)
    return int(vocab.content_start_id + ((base + 1) % int(vocab.base_vocab_size)))


def _prev_content_token(token_id: int, vocab: Vocab) -> int:
    base = int(token_id - vocab.content_start_id)
    return int(vocab.content_start_id + ((base - 1) % int(vocab.base_vocab_size)))


def _rank_marker(rank: int, group_size: int, vocab: Vocab) -> int:
    if rank <= 0:
        return int(vocab.rank_hi_token_id)
    if rank >= int(group_size) - 1:
        return int(vocab.rank_lo_token_id)
    return int(vocab.rank_mid_token_id)


def _build_candidate(
    *,
    topic_id: int,
    rank: int,
    group_size: int,
    vocab: Vocab,
    seq_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    prompt = [int(vocab.prompt_token_id), int(topic_id), int(vocab.separator_token_id)]
    marker = _rank_marker(rank, group_size, vocab)
    up_1 = _next_content_token(topic_id, vocab)
    up_2 = _next_content_token(up_1, vocab)
    down_1 = _prev_content_token(topic_id, vocab)
    down_2 = _prev_content_token(down_1, vocab)

    # Better-ranked samples follow topic->up trend; lower-ranked samples invert it.
    if rank <= int(group_size) // 2:
        response_content = [int(topic_id), int(up_1), int(up_2)]
    else:
        response_content = [int(topic_id), int(down_1), int(down_2)]
    response_tokens = [int(marker), *response_content, int(vocab.eos_id)]
    tokens = [*prompt, *response_tokens]

    input_ids = np.full((seq_length,), fill_value=int(vocab.pad_id), dtype=np.int64)
    labels = np.full((seq_length,), fill_value=int(vocab.ignore_index), dtype=np.int64)
    response_mask = np.zeros((seq_length,), dtype=np.float32)

    used = min(len(tokens), seq_length)
    input_ids[:used] = np.asarray(tokens[:used], dtype=np.int64)

    targets = [*response_content, int(vocab.eos_id)]
    start = len(prompt)
    for offset, target in enumerate(targets):
        pos = start + offset
        if pos >= seq_length:
            break
        labels[pos] = int(target)
        response_mask[pos] = 1.0

    # Descending scalar reward within each group.
    reward = float((int(group_size) - 1) - 2 * int(rank))
    return input_ids, labels, response_mask, reward


def _make_dataset_arrays(cfg: DataConfig, vocab: Vocab) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_prompts)
    g = int(cfg.group_size)
    t = int(cfg.seq_length)

    input_ids = np.empty((n, g, t), dtype=np.int64)
    labels = np.empty((n, g, t), dtype=np.int64)
    response_mask = np.empty((n, g, t), dtype=np.float32)
    group_rewards = np.empty((n, g), dtype=np.float32)

    for i in range(n):
        topic_offset = int(rng.integers(0, int(vocab.base_vocab_size)))
        topic_id = int(vocab.content_start_id + topic_offset)
        for r in range(g):
            ids_r, labels_r, mask_r, reward_r = _build_candidate(
                topic_id=topic_id,
                rank=r,
                group_size=g,
                vocab=vocab,
                seq_length=t,
            )
            input_ids[i, r] = ids_r
            labels[i, r] = labels_r
            response_mask[i, r] = mask_r
            group_rewards[i, r] = reward_r

    return input_ids, labels, response_mask, group_rewards


class SyntheticGrpoDataset:
    def __init__(
        self,
        *,
        input_ids: np.ndarray,
        labels: np.ndarray,
        response_mask: np.ndarray,
        group_rewards: np.ndarray,
        vocab: Vocab,
    ) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.response_mask = response_mask
        self.group_rewards = group_rewards
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        import torch

        ids = torch.tensor(self.input_ids[int(idx)], dtype=torch.long)
        labels = torch.tensor(self.labels[int(idx)], dtype=torch.long)
        response_mask = torch.tensor(self.response_mask[int(idx)], dtype=torch.float32)
        rewards = torch.tensor(self.group_rewards[int(idx)], dtype=torch.float32)
        return {
            "input_ids": ids,
            "attention_mask": (ids != int(self.vocab.pad_id)).to(torch.float32),
            "labels": labels,
            "response_mask": response_mask,
            "group_rewards": rewards,
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 8:
        raise ValueError("seq_length must be >= 8 for synthetic GRPO alignment sequences")
    if int(cfg.group_size) < 2:
        raise ValueError("group_size must be >= 2 for group-relative preference optimization")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    arrays = _make_dataset_arrays(cfg, vocab)
    ds = SyntheticGrpoDataset(
        input_ids=arrays[0],
        labels=arrays[1],
        response_mask=arrays[2],
        group_rewards=arrays[3],
        vocab=vocab,
    )

    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        out: dict[str, torch.Tensor] = {}
        for key in batch[0].keys():
            out[key] = torch.stack([row[key] for row in batch], dim=0)
        return out

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab


__all__ = ["DataConfig", "SyntheticGrpoDataset", "Vocab", "get_dataloaders"]
