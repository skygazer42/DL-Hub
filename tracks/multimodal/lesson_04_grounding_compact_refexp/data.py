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
_SIZES: tuple[str, ...] = ("small", "large")


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
    max_text_length: int = 8
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    min_objects: int = 2
    max_objects: int = 4


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "find",
        "locate",
        "point",
        "to",
        "the",
        "object",
        "at",
        "top",
        "bottom",
        "left",
        "right",
        "red",
        "green",
        "blue",
        "yellow",
        "square",
        "circle",
        "cross",
        "small",
        "large",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _cell_to_loc(row: int, col: int, grid_size: int) -> tuple[str, str]:
    vertical = "top" if row < int(grid_size) // 2 else "bottom"
    horizontal = "left" if col < int(grid_size) // 2 else "right"
    return vertical, horizontal


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
    h, w = int(image.shape[1]), int(image.shape[2])
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)

    ys = torch.arange(h, dtype=torch.float32).view(-1, 1).expand(h, w)
    xs = torch.arange(w, dtype=torch.float32).view(1, -1).expand(h, w)
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
    cell_size = float(int(cfg.image_size)) / float(int(cfg.grid_size))
    num_cells = int(cfg.grid_size) * int(cfg.grid_size)

    target_cell = int(torch.randint(0, num_cells, (1,), generator=generator).item())
    target_row = target_cell // int(cfg.grid_size)
    target_col = target_cell % int(cfg.grid_size)
    query_mode = int(idx) % 4

    color_name = tuple(_COLORS.keys())[int(torch.randint(0, len(_COLORS), (1,), generator=generator).item())]
    shape_name = _SHAPES[int(torch.randint(0, len(_SHAPES), (1,), generator=generator).item())]
    size_name = _SIZES[int(torch.randint(0, len(_SIZES), (1,), generator=generator).item())]

    chosen_cells = {target_cell}
    num_objects = int(torch.randint(int(cfg.min_objects), int(cfg.max_objects) + 1, (1,), generator=generator).item())
    objects: list[dict[str, object]] = []

    def _make_box(row: int, col: int, size_token: str) -> tuple[float, float, float, float, float, float]:
        dx = 0.2 + 0.6 * float(torch.rand(1, generator=generator).item())
        dy = 0.2 + 0.6 * float(torch.rand(1, generator=generator).item())
        cx = (float(col) + dx) * cell_size
        cy = (float(row) + dy) * cell_size
        side = (0.45 if size_token == "small" else 0.7) * cell_size
        width = side
        height = side
        x1 = max(0.0, cx - width / 2.0)
        y1 = max(0.0, cy - height / 2.0)
        x2 = min(float(cfg.image_size), cx + width / 2.0)
        y2 = min(float(cfg.image_size), cy + height / 2.0)
        return cx, cy, x1, y1, x2, y2

    target_cx, target_cy, x1, y1, x2, y2 = _make_box(target_row, target_col, size_name)
    target_loc = _cell_to_loc(target_row, target_col, int(cfg.grid_size))
    target = {
        "cell": target_cell,
        "row": target_row,
        "col": target_col,
        "color": color_name,
        "shape": shape_name,
        "size": size_name,
        "loc": target_loc,
        "cx": target_cx,
        "cy": target_cy,
        "box": [x1, y1, x2, y2],
    }
    objects.append(target)

    for _ in range(max(0, num_objects - 1)):
        while True:
            cell = int(torch.randint(0, num_cells, (1,), generator=generator).item())
            if cell in chosen_cells:
                continue
            row = cell // int(cfg.grid_size)
            col = cell % int(cfg.grid_size)
            candidate_color = tuple(_COLORS.keys())[
                int(torch.randint(0, len(_COLORS), (1,), generator=generator).item())
            ]
            candidate_shape = _SHAPES[int(torch.randint(0, len(_SHAPES), (1,), generator=generator).item())]
            candidate_size = _SIZES[int(torch.randint(0, len(_SIZES), (1,), generator=generator).item())]
            candidate_loc = _cell_to_loc(row, col, int(cfg.grid_size))

            if query_mode == 0 and candidate_color == color_name and candidate_shape == shape_name:
                continue
            if query_mode == 1 and candidate_color == color_name and candidate_shape == shape_name and candidate_size == size_name:
                continue
            if query_mode == 2 and candidate_color == color_name and candidate_shape == shape_name and candidate_size == size_name:
                continue
            if query_mode == 3 and candidate_color == color_name and candidate_loc == target_loc:
                continue
            break

        chosen_cells.add(cell)
        cx, cy, bx1, by1, bx2, by2 = _make_box(row, col, candidate_size)
        objects.append(
            {
                "cell": cell,
                "row": row,
                "col": col,
                "color": candidate_color,
                "shape": candidate_shape,
                "size": candidate_size,
                "loc": candidate_loc,
                "cx": cx,
                "cy": cy,
                "box": [bx1, by1, bx2, by2],
            }
        )

    if query_mode == 0:
        tokens = ["find", "the", color_name, shape_name]
    elif query_mode == 1:
        tokens = ["locate", "the", size_name, color_name, shape_name]
    elif query_mode == 2:
        tokens = ["point", "to", "the", size_name, color_name, shape_name]
    else:
        tokens = ["find", "the", color_name, "object", "at", target_loc[0], target_loc[1]]

    input_ids, attention_mask = vocab.encode(tokens, max_length=int(cfg.max_text_length))

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

    box_norm = torch.tensor(target["box"], dtype=torch.float32) / float(cfg.image_size)
    dx = (float(target["cx"]) / cell_size) - float(target_col)
    dy = (float(target["cy"]) / cell_size) - float(target_row)
    bw = (float(target["box"][2]) - float(target["box"][0])) / float(cfg.image_size)
    bh = (float(target["box"][3]) - float(target["box"][1])) / float(cfg.image_size)
    delta = torch.tensor([dx, dy, bw, bh], dtype=torch.float32).clamp(0.0, 1.0)

    return {
        "image": image.clamp(0.0, 1.0),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_cell": int(target_cell),
        "target_box": box_norm,
        "target_delta": delta,
        "query_text": " ".join(tokens),
    }


class SyntheticGroundingDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        if int(cfg.max_text_length) < 8:
            raise ValueError("max_text_length must be >= 8")
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
    dataset = SyntheticGroundingDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "input_ids": torch.stack([sample["input_ids"] for sample in batch], dim=0),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in batch], dim=0),
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


__all__ = ["DataConfig", "SyntheticGroundingDataset", "Vocab", "get_dataloaders"]
