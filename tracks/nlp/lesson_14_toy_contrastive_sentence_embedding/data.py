from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.toy_text import simple_tokenize


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_id: int
    unk_id: int

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode_tokens(self, tokens: list[str], *, max_length: int) -> tuple[list[int], list[int]]:
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens][: int(max_length)]
        attn = [1] * len(ids)
        while len(ids) < int(max_length):
            ids.append(self.pad_id)
            attn.append(0)
        return ids, attn

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "token_to_id": dict(self.token_to_id),
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 64
    max_length: int = 16
    dropout_prob: float = 0.2
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab(texts: list[str]) -> Vocab:
    id_to_token = ["<pad>", "<unk>"]
    token_to_id = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for token in simple_tokenize(text):
            if token in token_to_id:
                continue
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)


def _make_sentences(num_samples: int, seed: int) -> list[str]:
    rng = np.random.default_rng(int(seed))
    subjects = ["the student", "a robot", "my teammate", "this model", "our analyst"]
    verbs = ["writes", "tests", "reviews", "trains", "debugs", "summarizes"]
    objects = ["code", "sentences", "reports", "pipelines", "features", "datasets"]
    modifiers = ["carefully", "quickly", "daily", "at work", "in the lab", "for class"]
    templates = (
        "{subject} {verb} {objects}",
        "{subject} {verb} {objects} {mod}",
        "{subject} will {verb} {objects} {mod}",
    )

    out: list[str] = []
    for _ in range(int(num_samples)):
        out.append(
            str(rng.choice(templates)).format(
                subject=str(rng.choice(subjects)),
                verb=str(rng.choice(verbs)),
                objects=str(rng.choice(objects)),
                mod=str(rng.choice(modifiers)),
            )
        )
    return out


def _token_dropout(tokens: list[str], *, dropout_prob: float, rng: np.random.Generator) -> list[str]:
    if not tokens:
        return tokens

    keep_mask = rng.random(len(tokens)) >= float(dropout_prob)
    kept = [tok for tok, keep in zip(tokens, keep_mask) if bool(keep)]
    if kept:
        return kept
    return [tokens[int(rng.integers(0, len(tokens)))]]


class ContrastiveSentenceDataset:
    def __init__(self, *, texts: list[str], vocab: Vocab, cfg: DataConfig) -> None:
        self.texts = list(texts)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        import torch

        base_tokens = simple_tokenize(self.texts[int(idx)])
        rng1 = np.random.default_rng(int(self.cfg.seed) + int(idx) * 2 + 1)
        rng2 = np.random.default_rng(int(self.cfg.seed) + int(idx) * 2 + 2)
        view1_tokens = _token_dropout(base_tokens, dropout_prob=self.cfg.dropout_prob, rng=rng1)
        view2_tokens = _token_dropout(base_tokens, dropout_prob=self.cfg.dropout_prob, rng=rng2)

        view1_ids, view1_attn = self.vocab.encode_tokens(view1_tokens, max_length=int(self.cfg.max_length))
        view2_ids, view2_attn = self.vocab.encode_tokens(view2_tokens, max_length=int(self.cfg.max_length))

        return {
            "view1_input_ids": torch.tensor(view1_ids, dtype=torch.long),
            "view1_attention_mask": torch.tensor(view1_attn, dtype=torch.float32),
            "view2_input_ids": torch.tensor(view2_ids, dtype=torch.long),
            "view2_attention_mask": torch.tensor(view2_attn, dtype=torch.float32),
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    texts = _make_sentences(num_samples=int(cfg.num_samples), seed=int(cfg.seed))
    vocab = _build_vocab(texts)
    dataset = ContrastiveSentenceDataset(texts=texts, vocab=vocab, cfg=cfg)

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    def _collate(batch):
        return {
            "view1_input_ids": torch.stack([item["view1_input_ids"] for item in batch], dim=0),
            "view1_attention_mask": torch.stack([item["view1_attention_mask"] for item in batch], dim=0),
            "view2_input_ids": torch.stack([item["view2_input_ids"] for item in batch], dim=0),
            "view2_attention_mask": torch.stack([item["view2_attention_mask"] for item in batch], dim=0),
        }

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


__all__ = ["ContrastiveSentenceDataset", "DataConfig", "Vocab", "get_dataloaders"]
