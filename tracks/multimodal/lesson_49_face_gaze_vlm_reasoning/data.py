from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_QUERY_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("infer", "face", "gaze", "target", "left", "screen"),
    ("infer", "face", "gaze", "target", "right", "screen"),
    ("infer", "face", "gaze", "target", "upper", "screen"),
    ("infer", "face", "gaze", "target", "lower", "screen"),
)


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
            "token_to_id": {token: int(idx) for token, idx in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 16
    image_size: int = 48
    max_text_length: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "infer",
        "face",
        "gaze",
        "target",
        "left",
        "right",
        "upper",
        "lower",
        "screen",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _make_sample(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    face_cx = float(rng.uniform(0.40, 0.60))
    face_cy = float(rng.uniform(0.42, 0.62))
    face_w = float(rng.uniform(0.30, 0.38))
    face_h = float(rng.uniform(0.36, 0.44))
    eye_yaw = float(rng.uniform(-0.95, 0.95))
    eye_pitch = float(rng.uniform(-0.95, 0.95))

    axis = int(rng.integers(0, len(_QUERY_PATTERNS)))
    query_tokens = list(_QUERY_PATTERNS[axis])

    if query_tokens[4] == "left":
        query_bias = np.array([-0.18, 0.0], dtype=np.float32)
    elif query_tokens[4] == "right":
        query_bias = np.array([0.18, 0.0], dtype=np.float32)
    elif query_tokens[4] == "upper":
        query_bias = np.array([0.0, -0.18], dtype=np.float32)
    else:
        query_bias = np.array([0.0, 0.18], dtype=np.float32)

    gaze = np.array(
        [
            face_cx + 0.24 * eye_yaw + query_bias[0],
            face_cy + 0.24 * eye_pitch + query_bias[1],
        ],
        dtype=np.float32,
    )
    gaze = np.clip(gaze, 0.05, 0.95)

    return {
        "face_box": np.array(
            [
                face_cx - face_w * 0.5,
                face_cy - face_h * 0.5,
                face_cx + face_w * 0.5,
                face_cy + face_h * 0.5,
            ],
            dtype=np.float32,
        ),
        "eye_vector": np.array([eye_yaw, eye_pitch], dtype=np.float32),
        "target_gaze": gaze.astype(np.float32),
        "query_tokens": query_tokens,
    }


def _render_face(image_size: int, face_box: np.ndarray, eye_vector: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    x_norm = xx / float(max(image_size - 1, 1))
    y_norm = yy / float(max(image_size - 1, 1))

    x0, y0, x1, y1 = [float(v) for v in face_box]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    rx = max(0.08, 0.5 * (x1 - x0))
    ry = max(0.10, 0.5 * (y1 - y0))

    face_mask = (((x_norm - cx) / rx) ** 2 + ((y_norm - cy) / ry) ** 2 <= 1.0).astype(np.float32)
    image = np.full((image_size, image_size), 0.06, dtype=np.float32)
    image += 0.60 * face_mask

    eye_dx = 0.27 * rx
    eye_dy = -0.20 * ry
    pupil_shift_x = 0.040 * float(eye_vector[0])
    pupil_shift_y = 0.040 * float(eye_vector[1])
    for eye_sign in (-1.0, 1.0):
        eye_cx = cx + eye_sign * eye_dx
        eye_cy = cy + eye_dy
        eye_white = np.exp(
            -((x_norm - eye_cx) ** 2) / (2.0 * 0.018**2) - ((y_norm - eye_cy) ** 2) / (2.0 * 0.014**2)
        ).astype(np.float32)
        pupil = np.exp(
            -((x_norm - (eye_cx + pupil_shift_x)) ** 2) / (2.0 * 0.007**2)
            - ((y_norm - (eye_cy + pupil_shift_y)) ** 2) / (2.0 * 0.007**2)
        ).astype(np.float32)
        image += 0.16 * eye_white
        image -= 0.38 * pupil

    nose = np.exp(
        -((x_norm - cx) ** 2) / (2.0 * 0.016**2) - ((y_norm - (cy + 0.02 * ry)) ** 2) / (2.0 * 0.040**2)
    ).astype(np.float32)
    mouth = np.exp(
        -((x_norm - (cx + 0.03 * float(eye_vector[0]) * rx)) ** 2) / (2.0 * 0.06**2)
        - ((y_norm - (cy + 0.30 * ry)) ** 2) / (2.0 * 0.015**2)
    ).astype(np.float32)
    image -= 0.05 * nose
    image -= 0.09 * mouth
    image += 0.06 * float(eye_vector[0]) * np.clip((x_norm - cx) / rx, -1.0, 1.0) * face_mask
    return np.clip(image, 0.0, 1.0)


class SyntheticFaceGazeReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.max_text_length) < 8:
            raise ValueError("max_text_length must be >= 8")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = _make_sample(int(self.cfg.seed) * 1_000_003 + int(idx) * 211)
        query_tokens = [str(token) for token in sample["query_tokens"]]
        query_ids, query_mask = self.vocab.encode_tokens(query_tokens, max_length=int(self.cfg.max_text_length))
        image = _render_face(
            image_size=int(self.cfg.image_size),
            face_box=np.asarray(sample["face_box"], dtype=np.float32),
            eye_vector=np.asarray(sample["eye_vector"], dtype=np.float32),
        )
        return {
            "image": torch.from_numpy(image).to(torch.float32).unsqueeze(0),
            "face_box": torch.from_numpy(np.asarray(sample["face_box"], dtype=np.float32)).to(torch.float32),
            "eye_vector": torch.from_numpy(np.asarray(sample["eye_vector"], dtype=np.float32)).to(torch.float32),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_gaze": torch.from_numpy(np.asarray(sample["target_gaze"], dtype=np.float32)).to(torch.float32),
            "query_text": " ".join(query_tokens),
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticFaceGazeReasoningDataset(cfg, vocab=vocab)
    train_indices, val_indices = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader, vocab


__all__ = [
    "DataConfig",
    "SyntheticFaceGazeReasoningDataset",
    "Vocab",
    "get_dataloaders",
]
