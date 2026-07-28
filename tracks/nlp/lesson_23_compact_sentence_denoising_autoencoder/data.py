from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.synthetic_text import simple_tokenize


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int
    mask_id: int

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode_source(self, text: str, *, max_length: int) -> tuple[list[int], list[int]]:
        tokens = simple_tokenize(text)
        ids = [
            self.mask_id if token == "mask" else self.token_to_id.get(token, self.unk_id)
            for token in tokens
        ][: int(max_length)]
        mask = [1] * len(ids)
        while len(ids) < int(max_length):
            ids.append(self.pad_id)
            mask.append(0)
        return ids, mask

    def build_target(self, text: str, *, max_length: int) -> tuple[list[int], list[int]]:
        tokens = simple_tokenize(text)[: int(max_length)]
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        tgt_in = [self.bos_id] + token_ids
        tgt_out = token_ids + [self.eos_id]
        full_length = int(max_length) + 1
        while len(tgt_in) < full_length:
            tgt_in.append(self.pad_id)
        while len(tgt_out) < full_length:
            tgt_out.append(self.pad_id)
        return tgt_in[:full_length], tgt_out[:full_length]

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "mask_id": self.mask_id,
            "token_to_id": dict(self.token_to_id),
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 64
    max_length: int = 12
    corruption_prob: float = 0.35
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
    return [
        rng.choice(templates).format(
            subject=rng.choice(subjects),
            verb=rng.choice(verbs),
            object=rng.choice(objects),
            adverb=rng.choice(adverbs),
            place=rng.choice(places),
        )
        for _ in range(int(num_samples))
    ]


def _build_vocab(texts: list[str]) -> Vocab:
    id_to_token = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]
    token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
    tokens: list[str] = []
    for text in texts:
        tokens.extend(simple_tokenize(text))
    for token in sorted(set(tokens)):
        if token not in token_to_id:
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
    return Vocab(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        mask_id=4,
    )


def _corrupt_text(text: str, *, vocab: Vocab, corruption_prob: float, seed: int) -> str:
    rng = np.random.default_rng(int(seed))
    tokens = simple_tokenize(text)
    corrupted: list[str] = []
    for token in tokens:
        if rng.random() < float(corruption_prob):
            mode = int(rng.integers(0, 3))
            if mode == 0:
                corrupted.append("<mask>")
            elif mode == 1:
                continue
            else:
                candidates = [
                    word
                    for word in vocab.id_to_token[5:]
                    if word != token
                ]
                corrupted.append(str(rng.choice(candidates)) if candidates else token)
        else:
            corrupted.append(token)
    if not corrupted:
        corrupted = ["<mask>"]
    if len(corrupted) >= 2 and rng.random() < 0.35:
        idx = int(rng.integers(0, len(corrupted) - 1))
        corrupted[idx], corrupted[idx + 1] = corrupted[idx + 1], corrupted[idx]
    return " ".join(corrupted)


class SentenceDenoisingDataset:
    def __init__(self, *, texts: list[str], vocab: Vocab, cfg: DataConfig) -> None:
        self.texts = list(texts)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        import torch

        clean_text = self.texts[int(idx)]
        corrupt_text = _corrupt_text(
            clean_text,
            vocab=self.vocab,
            corruption_prob=float(self.cfg.corruption_prob),
            seed=int(self.cfg.seed) * 991 + int(idx),
        )
        src_ids, src_mask = self.vocab.encode_source(corrupt_text, max_length=int(self.cfg.max_length))
        tgt_in_ids, tgt_out_ids = self.vocab.build_target(clean_text, max_length=int(self.cfg.max_length))
        tgt_mask = [1 if token != self.vocab.pad_id else 0 for token in tgt_in_ids]
        return (
            {
                "src_ids": torch.tensor(src_ids, dtype=torch.long),
                "src_mask": torch.tensor(src_mask, dtype=torch.float32),
                "tgt_in_ids": torch.tensor(tgt_in_ids, dtype=torch.long),
                "tgt_mask": torch.tensor(tgt_mask, dtype=torch.float32),
            },
            {"tgt_out_ids": torch.tensor(tgt_out_ids, dtype=torch.long)},
        )


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    texts = _make_sentences(num_samples=int(cfg.num_samples), seed=int(cfg.seed))
    vocab = _build_vocab(texts)
    dataset = SentenceDenoisingDataset(texts=texts, vocab=vocab, cfg=cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch):
        input_keys = batch[0][0].keys()
        target_keys = batch[0][1].keys()
        inputs = {key: torch.stack([item[0][key] for item in batch], dim=0) for key in input_keys}
        targets = {key: torch.stack([item[1][key] for item in batch], dim=0) for key in target_keys}
        return inputs, targets

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab


__all__ = ["DataConfig", "Vocab", "get_dataloaders"]
