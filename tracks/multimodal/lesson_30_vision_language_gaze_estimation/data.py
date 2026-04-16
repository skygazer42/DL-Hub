from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_DIRECTIONS: tuple[str, ...] = ("left", "right", "up", "down")


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id[self.pad_token])

    @property
    def bos_id(self) -> int:
        return int(self.token_to_id[self.bos_token])

    @property
    def eos_id(self) -> int:
        return int(self.token_to_id[self.eos_token])

    @property
    def size(self) -> int:
        return int(len(self.id_to_token))

    def encode_tokens(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = [self.bos_id, *[int(self.token_to_id[token]) for token in tokens], self.eos_id]
        if len(seq) > int(max_length):
            raise ValueError(f"Sequence exceeds max_length={int(max_length)}")
        pad_count = int(max_length) - len(seq)
        seq.extend([self.pad_id] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
        return torch.tensor(seq, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": int(self.pad_id),
            "bos_id": int(self.bos_id),
            "eos_id": int(self.eos_id),
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 16
    image_size: int = 24
    heatmap_size: int = 12
    max_text_length: int = 8
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "person",
        "looks",
        "toward",
        "left",
        "right",
        "up",
        "down",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _render_image(*, image_size: int, head_xy: torch.Tensor, gaze_xy: torch.Tensor) -> torch.Tensor:
    size = int(image_size)
    image = torch.full((3, size, size), 0.06, dtype=torch.float32)
    image[:, 0, :] = 0.1
    image[:, -1, :] = 0.1
    image[:, :, 0] = 0.1
    image[:, :, -1] = 0.1

    ys = torch.arange(size, dtype=torch.float32).view(-1, 1).expand(size, size)
    xs = torch.arange(size, dtype=torch.float32).view(1, -1).expand(size, size)
    head_x = float(head_xy[0].item()) * float(size - 1)
    head_y = float(head_xy[1].item()) * float(size - 1)
    gaze_x = float(gaze_xy[0].item()) * float(size - 1)
    gaze_y = float(gaze_xy[1].item()) * float(size - 1)

    head_mask = (xs - head_x).pow(2) + (ys - head_y).pow(2) <= 5.5
    gaze_mask = (xs - gaze_x).pow(2) + (ys - gaze_y).pow(2) <= 4.5
    image[0] = torch.where(head_mask, torch.tensor(0.95), image[0])
    image[1] = torch.where(head_mask, torch.tensor(0.25), image[1])
    image[2] = torch.where(head_mask, torch.tensor(0.25), image[2])
    image[0] = torch.where(gaze_mask, torch.tensor(0.25), image[0])
    image[1] = torch.where(gaze_mask, torch.tensor(0.95), image[1])
    image[2] = torch.where(gaze_mask, torch.tensor(0.25), image[2])
    return image.clamp(0.0, 1.0)


def _make_heatmap(*, heatmap_size: int, target_xy: torch.Tensor) -> torch.Tensor:
    size = int(heatmap_size)
    ys = torch.arange(size, dtype=torch.float32).view(-1, 1).expand(size, size)
    xs = torch.arange(size, dtype=torch.float32).view(1, -1).expand(size, size)
    tx = float(target_xy[0].item()) * float(size - 1)
    ty = float(target_xy[1].item()) * float(size - 1)
    sigma = max(1.0, float(size) / 8.0)
    heat = torch.exp(-((xs - tx).pow(2) + (ys - ty).pow(2)) / (2.0 * sigma * sigma))
    heat = heat / heat.max().clamp_min(1e-6)
    return heat.unsqueeze(0)


def _sample_record(*, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, str]:
    head_x = 0.45 + 0.1 * float(torch.rand(1, generator=generator).item())
    head_y = 0.45 + 0.1 * float(torch.rand(1, generator=generator).item())
    direction = _DIRECTIONS[int(torch.randint(0, len(_DIRECTIONS), (1,), generator=generator).item())]
    offset = 0.32 + 0.05 * float(torch.rand(1, generator=generator).item())

    gaze_x, gaze_y = head_x, head_y
    if direction == "left":
        gaze_x = head_x - offset
    elif direction == "right":
        gaze_x = head_x + offset
    elif direction == "up":
        gaze_y = head_y - offset
    else:
        gaze_y = head_y + offset

    gaze_x = max(0.05, min(0.95, gaze_x))
    gaze_y = max(0.05, min(0.95, gaze_y))
    head_xy = torch.tensor([head_x, head_y], dtype=torch.float32)
    gaze_xy = torch.tensor([gaze_x, gaze_y], dtype=torch.float32)
    return head_xy, gaze_xy, direction


class ToyGazeEstimationDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        self.generator = torch.Generator().manual_seed(int(cfg.seed))
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.heatmap_size) < 8:
            raise ValueError("heatmap_size must be >= 8")
        if int(cfg.max_text_length) < 6:
            raise ValueError("max_text_length must be >= 6")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        del idx
        head_xy, gaze_xy, direction = _sample_record(generator=self.generator)
        prompt_tokens = ["person", "looks", "toward", direction]
        input_ids, attention_mask = self.vocab.encode_tokens(
            prompt_tokens, max_length=int(self.cfg.max_text_length)
        )
        image = _render_image(
            image_size=int(self.cfg.image_size),
            head_xy=head_xy,
            gaze_xy=gaze_xy,
        )
        target_heatmap = _make_heatmap(heatmap_size=int(self.cfg.heatmap_size), target_xy=gaze_xy)
        return {
            "image": image,
            "head_xy": head_xy,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_point": gaze_xy,
            "target_heatmap": target_heatmap,
            "prompt_text": " ".join(prompt_tokens),
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = ToyGazeEstimationDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "head_xy": torch.stack([sample["head_xy"] for sample in batch], dim=0),
            "input_ids": torch.stack([sample["input_ids"] for sample in batch], dim=0),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in batch], dim=0),
            "target_point": torch.stack([sample["target_point"] for sample in batch], dim=0),
            "target_heatmap": torch.stack([sample["target_heatmap"] for sample in batch], dim=0),
            "prompt_text": [str(sample["prompt_text"]) for sample in batch],
        }

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
