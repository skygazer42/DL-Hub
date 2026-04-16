from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class PlateVocab:
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
    num_samples: int = 2048
    batch_size: int = 32
    image_height: int = 24
    image_width: int = 72
    plate_length: int = 6
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    noise_std: float = 0.04


def _char_template(char_id: int) -> np.ndarray:
    rng = np.random.default_rng(int(char_id) * 7919)
    template = (rng.random((7, 5)) > 0.45).astype(np.float32)
    template[0, :] = (char_id % 2)
    template[-1, :] = ((char_id // 2) % 2)
    template[:, 0] = ((char_id // 3) % 2)
    template[:, -1] = ((char_id // 5) % 2)
    template[3, 2] = 1.0
    return template


class SyntheticLicensePlateDataset:
    """Render fixed-length synthetic plates with deterministic character templates."""

    def __init__(self, cfg: DataConfig, vocab: PlateVocab | None = None) -> None:
        if int(cfg.image_height) < 16 or int(cfg.image_width) < 48:
            raise ValueError("image geometry is too small for license plate rendering")
        if int(cfg.plate_length) < 4:
            raise ValueError("plate_length must be >= 4")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg
        self.vocab = PlateVocab() if vocab is None else vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        h = int(self.cfg.image_height)
        w = int(self.cfg.image_width)
        plate_length = int(self.cfg.plate_length)
        rng = np.random.default_rng(int(self.cfg.seed) * 3_000_001 + int(idx))

        labels = rng.integers(1, self.vocab.size, size=plate_length, dtype=np.int64)
        image = np.full((h, w), 0.05, dtype=np.float32)
        image[2 : h - 2, 2 : w - 2] = 0.82

        cell_w = (w - 4) // plate_length
        for position, char_id in enumerate(labels.tolist()):
            template = _char_template(int(char_id))
            scale_y = 2
            scale_x = max(1, (cell_w - 2) // template.shape[1])
            glyph = np.kron(template, np.ones((scale_y, scale_x), dtype=np.float32))
            glyph_h, glyph_w = glyph.shape
            start_y = max(3, (h - glyph_h) // 2 + int(rng.integers(-1, 2)))
            cell_x0 = 2 + position * cell_w
            start_x = cell_x0 + max(1, (cell_w - glyph_w) // 2 + int(rng.integers(-1, 2)))
            end_y = min(h - 2, start_y + glyph_h)
            end_x = min(w - 2, start_x + glyph_w)
            crop = glyph[: end_y - start_y, : end_x - start_x]
            image[start_y:end_y, start_x:end_x] -= 0.58 * crop
            if position < plate_length - 1:
                sep_x = cell_x0 + cell_w - 1
                image[4 : h - 4, sep_x : sep_x + 1] -= 0.08

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(h, w)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(labels)


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    vocab = PlateVocab()
    dataset = SyntheticLicensePlateDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.stack([item[1] for item in batch], dim=0)
        return images, labels

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


__all__ = ["DataConfig", "PlateVocab", "SyntheticLicensePlateDataset", "get_dataloaders"]
