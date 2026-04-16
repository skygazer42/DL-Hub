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
    mask_id: int

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, *, max_length: int) -> tuple[list[int], list[int]]:
        tokens = simple_tokenize(text)
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens][: int(max_length)]
        attn = [1] * len(ids)
        while len(ids) < int(max_length):
            ids.append(self.pad_id)
            attn.append(0)
        return ids, attn

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "mask_id": self.mask_id,
            "token_to_id": dict(self.token_to_id),
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 64
    max_length: int = 16
    mask_prob: float = 0.15
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _make_sentences(num_samples: int, seed: int) -> list[str]:
    rng = np.random.default_rng(int(seed))
    subjects = ["the cat", "a student", "my friend", "the robot", "our team", "this model"]
    verbs = ["likes", "builds", "tests", "reads", "fixes", "studies", "trains"]
    objects = ["code", "books", "music", "pipelines", "datasets", "tokens", "features"]
    adverbs = ["carefully", "quickly", "daily", "happily", "quietly", "eagerly"]
    places = ["at home", "in class", "at work", "in the lab", "after lunch", "on weekends"]
    templates = [
        "{subject} {verb} {object}",
        "{subject} {verb} {object} {adverb}",
        "{subject} {verb} {object} {place}",
        "{subject} {verb} {object} {adverb} {place}",
    ]

    sentences: list[str] = []
    for _ in range(int(num_samples)):
        sentence = rng.choice(templates).format(
            subject=rng.choice(subjects),
            verb=rng.choice(verbs),
            object=rng.choice(objects),
            adverb=rng.choice(adverbs),
            place=rng.choice(places),
        )
        sentences.append(sentence)
    return sentences


def _build_vocab(texts: list[str]) -> Vocab:
    tokens: list[str] = []
    for text in texts:
        tokens.extend(simple_tokenize(text))

    id_to_token = ["<pad>", "<unk>", "<mask>"]
    token_to_id = {"<pad>": 0, "<unk>": 1, "<mask>": 2}
    for token in sorted(set(tokens)):
        if token in token_to_id:
            continue
        token_to_id[token] = len(id_to_token)
        id_to_token.append(token)
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1, mask_id=2)


class MaskedLanguageModelingDataset:
    def __init__(self, *, texts: list[str], vocab: Vocab, cfg: DataConfig) -> None:
        self.texts = list(texts)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        import torch

        text = self.texts[int(idx)]
        token_ids, attention_mask = self.vocab.encode(text, max_length=int(self.cfg.max_length))

        rng = np.random.default_rng(int(self.cfg.seed) + int(idx))
        masked_positions = [False] * len(token_ids)
        labels = [-100] * len(token_ids)
        valid_positions = [i for i, flag in enumerate(attention_mask) if flag == 1]

        for pos in valid_positions:
            if rng.random() < float(self.cfg.mask_prob):
                masked_positions[pos] = True

        if valid_positions and not any(masked_positions):
            forced_pos = int(rng.choice(valid_positions))
            masked_positions[forced_pos] = True

        for pos, is_masked in enumerate(masked_positions):
            if is_masked and attention_mask[pos] == 1:
                labels[pos] = token_ids[pos]
                token_ids[pos] = self.vocab.mask_id

        inputs = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "masked_positions": torch.tensor(masked_positions, dtype=torch.bool),
        }
        targets = {"labels": torch.tensor(labels, dtype=torch.long)}
        return inputs, targets


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    texts = _make_sentences(num_samples=int(cfg.num_samples), seed=int(cfg.seed))
    vocab = _build_vocab(texts)
    dataset = MaskedLanguageModelingDataset(texts=texts, vocab=vocab, cfg=cfg)

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    def _collate(batch):
        input_ids = torch.stack([item[0]["input_ids"] for item in batch], dim=0)
        attention_mask = torch.stack([item[0]["attention_mask"] for item in batch], dim=0)
        masked_positions = torch.stack([item[0]["masked_positions"] for item in batch], dim=0)
        labels = torch.stack([item[1]["labels"] for item in batch], dim=0)
        return (
            {"input_ids": input_ids, "attention_mask": attention_mask, "masked_positions": masked_positions},
            {"labels": labels},
        )

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


__all__ = ["DataConfig", "MaskedLanguageModelingDataset", "Vocab", "get_dataloaders"]
