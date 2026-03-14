from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.15, 0.15),
    "green": (0.15, 0.95, 0.2),
    "blue": (0.2, 0.35, 1.0),
    "yellow": (1.0, 0.9, 0.15),
}
_SHAPES: tuple[str, ...] = ("square", "circle", "cross")


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id[self.pad_token])

    @property
    def size(self) -> int:
        return int(len(self.id_to_token))

    def encode(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = [int(self.token_to_id[token]) for token in tokens]
        if len(token_ids) > int(max_length):
            raise ValueError(f"Sequence exceeds max_length={int(max_length)}.")
        pad_count = int(max_length) - len(token_ids)
        token_ids.extend([self.pad_id] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": int(self.pad_id),
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 32
    image_size: int = 32
    grid_size: int = 4
    max_text_length: int = 6
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    min_objects: int = 2
    max_objects: int = 4


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "detect",
        "find",
        "red",
        "green",
        "blue",
        "yellow",
        "square",
        "circle",
        "cross",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _render_shape(
    *,
    image: torch.Tensor,
    color_name: str,
    shape_name: str,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> None:
    image_h, image_w = int(image.shape[1]), int(image.shape[2])
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)

    ys = torch.arange(image_h, dtype=torch.float32).view(-1, 1).expand(image_h, image_w)
    xs = torch.arange(image_w, dtype=torch.float32).view(1, -1).expand(image_h, image_w)
    half_w = max(1.0, float(width) / 2.0)
    half_h = max(1.0, float(height) / 2.0)

    if shape_name == "square":
        mask = (xs - float(center_x)).abs() <= half_w
        mask = mask & ((ys - float(center_y)).abs() <= half_h)
    elif shape_name == "circle":
        rx = half_w
        ry = half_h
        mask = ((xs - float(center_x)) / rx).pow(2) + ((ys - float(center_y)) / ry).pow(2) <= 1.0
    elif shape_name == "cross":
        thickness = max(1.0, min(half_w, half_h) / 2.0)
        vertical = (xs - float(center_x)).abs() <= thickness
        vertical = vertical & ((ys - float(center_y)).abs() <= half_h)
        horizontal = (ys - float(center_y)).abs() <= thickness
        horizontal = horizontal & ((xs - float(center_x)).abs() <= half_w)
        mask = vertical | horizontal
    else:
        raise ValueError(f"Unsupported shape: {shape_name}")

    image[:] = torch.where(mask.unsqueeze(0), color.expand_as(image), image)


def _build_record(
    *,
    cfg: DataConfig,
    idx: int,
    vocab: Vocab,
    generator: torch.Generator,
) -> dict[str, object]:
    if int(cfg.max_objects) < int(cfg.min_objects):
        raise ValueError("max_objects must be >= min_objects")

    cell_size = float(int(cfg.image_size)) / float(int(cfg.grid_size))
    num_cells = int(cfg.grid_size) * int(cfg.grid_size)
    num_objects = int(
        torch.randint(
            int(cfg.min_objects),
            int(cfg.max_objects) + 1,
            (1,),
            generator=generator,
        ).item()
    )

    chosen_cells: set[int] = set()
    used_pairs: set[tuple[str, str]] = set()
    objects: list[dict[str, object]] = []

    def _make_box(row: int, col: int) -> tuple[float, float, float, float, float, float]:
        dx = 0.2 + 0.6 * float(torch.rand(1, generator=generator).item())
        dy = 0.2 + 0.6 * float(torch.rand(1, generator=generator).item())
        center_x = (float(col) + dx) * cell_size
        center_y = (float(row) + dy) * cell_size
        side = (0.5 + 0.18 * float(torch.rand(1, generator=generator).item())) * cell_size
        x1 = max(0.0, center_x - side / 2.0)
        y1 = max(0.0, center_y - side / 2.0)
        x2 = min(float(cfg.image_size), center_x + side / 2.0)
        y2 = min(float(cfg.image_size), center_y + side / 2.0)
        return center_x, center_y, x1, y1, x2, y2

    while len(objects) < num_objects:
        cell = int(torch.randint(0, num_cells, (1,), generator=generator).item())
        if cell in chosen_cells:
            continue
        color_name = tuple(_COLORS.keys())[int(torch.randint(0, len(_COLORS), (1,), generator=generator).item())]
        shape_name = _SHAPES[int(torch.randint(0, len(_SHAPES), (1,), generator=generator).item())]
        pair = (color_name, shape_name)
        if pair in used_pairs:
            continue

        row = cell // int(cfg.grid_size)
        col = cell % int(cfg.grid_size)
        center_x, center_y, x1, y1, x2, y2 = _make_box(row, col)
        chosen_cells.add(cell)
        used_pairs.add(pair)
        objects.append(
            {
                "cell": cell,
                "row": row,
                "col": col,
                "color": color_name,
                "shape": shape_name,
                "cx": center_x,
                "cy": center_y,
                "box": [x1, y1, x2, y2],
            }
        )

    positive = int(idx) % 2 == 0
    query_prefix = "detect" if int(idx) % 4 < 2 else "find"

    if positive:
        target_obj = objects[int(torch.randint(0, len(objects), (1,), generator=generator).item())]
        query_color = str(target_obj["color"])
        query_shape = str(target_obj["shape"])
        target_present = 1.0
        target_cell = int(target_obj["cell"])
        target_box = torch.tensor(target_obj["box"], dtype=torch.float32) / float(cfg.image_size)
        dx = (float(target_obj["cx"]) / cell_size) - float(target_obj["col"])
        dy = (float(target_obj["cy"]) / cell_size) - float(target_obj["row"])
        bw = (float(target_obj["box"][2]) - float(target_obj["box"][0])) / float(cfg.image_size)
        bh = (float(target_obj["box"][3]) - float(target_obj["box"][1])) / float(cfg.image_size)
        target_delta = torch.tensor([dx, dy, bw, bh], dtype=torch.float32).clamp(0.0, 1.0)
    else:
        all_pairs = [(c, s) for c in _COLORS for s in _SHAPES]
        absent_pairs = [pair for pair in all_pairs if pair not in used_pairs]
        query_color, query_shape = absent_pairs[
            int(torch.randint(0, len(absent_pairs), (1,), generator=generator).item())
        ]
        target_present = 0.0
        target_cell = 0
        target_box = torch.zeros(4, dtype=torch.float32)
        target_delta = torch.zeros(4, dtype=torch.float32)

    query_tokens = [query_prefix, query_color, query_shape]
    query_ids, attention_mask = vocab.encode(query_tokens, max_length=int(cfg.max_text_length))

    image = torch.full((3, int(cfg.image_size), int(cfg.image_size)), 0.03, dtype=torch.float32)
    for obj in objects:
        _render_shape(
            image=image,
            color_name=str(obj["color"]),
            shape_name=str(obj["shape"]),
            center_x=float(obj["cx"]),
            center_y=float(obj["cy"]),
            width=float(obj["box"][2]) - float(obj["box"][0]),
            height=float(obj["box"][3]) - float(obj["box"][1]),
        )

    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08

    return {
        "image": image.clamp(0.0, 1.0),
        "query_ids": query_ids,
        "attention_mask": attention_mask,
        "target_present": torch.tensor(float(target_present), dtype=torch.float32),
        "target_cell": int(target_cell),
        "target_box": target_box,
        "target_delta": target_delta,
        "query_text": " ".join(query_tokens),
    }


class ToyOwlVitDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        if int(cfg.max_text_length) < 3:
            raise ValueError("max_text_length must be >= 3")
        if int(cfg.grid_size) < 2:
            raise ValueError("grid_size must be >= 2")
        if int(cfg.image_size) < int(cfg.grid_size) * 4:
            raise ValueError("image_size is too small for the requested grid_size")
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.records = [
            _build_record(cfg=cfg, idx=idx, vocab=vocab, generator=generator)
            for idx in range(int(cfg.num_samples))
        ]

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.records[int(idx)]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = ToyOwlVitDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "query_ids": torch.stack([sample["query_ids"] for sample in batch], dim=0),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in batch], dim=0),
            "target_present": torch.stack([sample["target_present"] for sample in batch], dim=0),
            "target_cell": torch.tensor([sample["target_cell"] for sample in batch], dtype=torch.long),
            "target_box": torch.stack([sample["target_box"] for sample in batch], dim=0),
            "target_delta": torch.stack([sample["target_delta"] for sample in batch], dim=0),
            "query_text": [str(sample["query_text"]) for sample in batch],
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


__all__ = ["DataConfig", "ToyOwlVitDataset", "Vocab", "get_dataloaders"]
