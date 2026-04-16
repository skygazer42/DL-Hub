from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class SpottingVocab:
    alphabet: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pad_id: int = 0

    @property
    def size(self) -> int:
        return 1 + len(self.alphabet)

    @property
    def id_to_token(self) -> list[str]:
        return ["<pad>"] + list(self.alphabet)

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.id_to_token)}

    def to_dict(self) -> dict[str, object]:
        return {"pad_id": int(self.pad_id), "alphabet": self.alphabet, "token_to_id": self.token_to_id}


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1536
    batch_size: int = 16
    image_size: int = 40
    text_length: int = 4
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    noise_std: float = 0.02


def _char_template(char_id: int) -> np.ndarray:
    rng = np.random.default_rng(13_357 + int(char_id) * 10_003)
    template = (rng.random((7, 5)) > 0.48).astype(np.float32)
    template[0, :] = (char_id % 2)
    template[-1, :] = ((char_id // 2) % 2)
    template[:, 0] = ((char_id // 3) % 2)
    template[:, -1] = ((char_id // 5) % 2)
    template[3, 2] = 1.0
    return template


class SyntheticSceneTextSpottingDataset:
    """Generate toy images with one text region and a short recognition target."""

    def __init__(self, cfg: DataConfig, vocab: SpottingVocab | None = None) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.text_length) < 3:
            raise ValueError("text_length must be >= 3")
        if int(cfg.in_channels) != 1:
            raise ValueError("this toy lesson expects in_channels == 1")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg
        self.vocab = SpottingVocab() if vocab is None else vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        text_length = int(self.cfg.text_length)
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))

        image = np.full((size, size), 0.92, dtype=np.float32)
        image += rng.normal(0.0, 0.01, size=(size, size)).astype(np.float32)

        box_h = int(rng.integers(max(12, size // 4), max(13, size // 2)))
        box_w = int(rng.integers(max(16, size // 2), max(17, size - 6)))
        y0 = int(rng.integers(2, max(3, size - box_h - 2)))
        x0 = int(rng.integers(2, max(3, size - box_w - 2)))
        y1 = min(size - 1, y0 + box_h)
        x1 = min(size - 1, x0 + box_w)

        image[y0:y1, x0:x1] = 0.85
        image[y0 : y0 + 1, x0:x1] = 0.72
        image[y1 - 1 : y1, x0:x1] = 0.72
        image[y0:y1, x0 : x0 + 1] = 0.72
        image[y0:y1, x1 - 1 : x1] = 0.72

        text_tokens = rng.integers(1, self.vocab.size, size=text_length, dtype=np.int64)
        score_map = np.zeros((1, size, size), dtype=np.float32)
        score_map[0, y0:y1, x0:x1] = 1.0

        cell_w = max(4, (x1 - x0 - 2) // text_length)
        baseline = y0 + (y1 - y0) // 2 + 4
        for pos, token in enumerate(text_tokens.tolist()):
            template = _char_template(int(token))
            scale_y = 2
            scale_x = max(1, (cell_w - 2) // template.shape[1])
            glyph = np.kron(template, np.ones((scale_y, scale_x), dtype=np.float32))
            gh, gw = glyph.shape
            start_y = int(np.clip(baseline - gh + int(rng.integers(-1, 2)), y0 + 1, y1 - gh - 1))
            cell_x0 = x0 + 1 + pos * cell_w
            start_x = int(np.clip(cell_x0 + (cell_w - gw) // 2, x0 + 1, x1 - gw - 1))
            image[start_y : start_y + gh, start_x : start_x + gw] -= 0.72 * glyph

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        target = {
            "score_map": torch.from_numpy(score_map),
            "text_tokens": torch.from_numpy(text_tokens),
            "first_token": torch.tensor(int(text_tokens[0]), dtype=torch.long),
        }
        return torch.from_numpy(image).unsqueeze(0), target


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    vocab = SpottingVocab()
    dataset = SyntheticSceneTextSpottingDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        score_map = torch.stack([item[1]["score_map"] for item in batch], dim=0)
        text_tokens = torch.stack([item[1]["text_tokens"] for item in batch], dim=0)
        first_token = torch.stack([item[1]["first_token"] for item in batch], dim=0)
        targets = {"score_map": score_map, "text_tokens": text_tokens, "first_token": first_token}
        return images, targets

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
    return train_loader, val_loader


__all__ = ["DataConfig", "SpottingVocab", "SyntheticSceneTextSpottingDataset", "get_dataloaders"]
