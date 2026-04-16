from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.toy_text import Vocab, simple_tokenize


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    batch_size: int = 16
    max_length: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


_SCENARIOS = (
    ("book", "on", "table"),
    ("dog", "near", "door"),
    ("teacher", "inside", "classroom"),
    ("chef", "in", "kitchen"),
    ("runner", "on", "track"),
    ("bird", "above", "tree"),
    ("child", "beside", "window"),
    ("cat", "under", "chair"),
)

_MODIFIERS = (
    "quickly",
    "carefully",
    "today",
    "outside",
    "indoors",
    "silently",
)


def _build_vocab(texts: list[str]) -> Vocab:
    token_to_id = {"<pad>": 0, "<unk>": 1}
    id_to_token = ["<pad>", "<unk>"]
    for text in texts:
        for token in simple_tokenize(text):
            if token in token_to_id:
                continue
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)


def _encode(vocab: Vocab, text: str, *, max_length: int) -> tuple[list[int], list[int]]:
    ids = [vocab.token_to_id.get(tok, vocab.unk_id) for tok in simple_tokenize(text)[: int(max_length)]]
    attention = [1] * len(ids)
    while len(ids) < int(max_length):
        ids.append(vocab.pad_id)
        attention.append(0)
    return ids, attention


def _jaccard(tokens_a: list[str], tokens_b: list[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return float(len(set_a & set_b) / len(union))


def _make_pair(rng: np.random.Generator) -> tuple[str, float]:
    subj, rel, obj = _SCENARIOS[int(rng.integers(0, len(_SCENARIOS)))]
    variant = int(rng.integers(0, 4))

    sent_a_tokens = ["sentence_a", subj, "is", rel, "the", obj]
    sent_a_tokens.append(str(rng.choice(_MODIFIERS)))

    if variant == 0:
        sent_b_tokens = ["sentence_b", subj, "is", rel, "the", obj, str(rng.choice(_MODIFIERS))]
    elif variant == 1:
        alt_obj = str(rng.choice([x[2] for x in _SCENARIOS if x[2] != obj]))
        sent_b_tokens = ["sentence_b", subj, "is", rel, "the", alt_obj, str(rng.choice(_MODIFIERS))]
    elif variant == 2:
        alt_subj = str(rng.choice([x[0] for x in _SCENARIOS if x[0] != subj]))
        sent_b_tokens = ["sentence_b", alt_subj, "is", "near", "the", obj, str(rng.choice(_MODIFIERS))]
    else:
        sent_b_tokens = ["sentence_b", "someone", "mentions", obj, "briefly"]

    similarity = _jaccard(sent_a_tokens[1:], sent_b_tokens[1:])
    noise = float(rng.normal(0.0, 0.02))
    score = float(np.clip(similarity + noise, 0.0, 1.0))
    text = " ".join(sent_a_tokens + sent_b_tokens)
    return text, score


def _make_examples(config: DataConfig) -> list[tuple[str, float]]:
    rng = np.random.default_rng(int(config.seed))
    examples: list[tuple[str, float]] = []
    for _ in range(int(config.num_samples)):
        examples.append(_make_pair(rng))
    rng.shuffle(examples)
    return examples


class SemanticTextualSimilarityDataset:
    def __init__(self, *, examples: list[tuple[str, float]], vocab: Vocab, cfg: DataConfig) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        text, score = self.examples[int(idx)]
        input_ids, attention_mask = _encode(self.vocab, text, max_length=int(self.cfg.max_length))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "scores": torch.tensor(score, dtype=torch.float32),
        }


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(config)
    vocab = _build_vocab([text for text, _ in examples])
    dataset = SemanticTextualSimilarityDataset(examples=examples, vocab=vocab, cfg=config)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(config.val_fraction),
        seed=int(config.seed),
    )

    def _collate(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "scores": torch.stack([item["scores"] for item in batch], dim=0),
        }

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab


__all__ = ["DataConfig", "SemanticTextualSimilarityDataset", "get_dataloaders"]
