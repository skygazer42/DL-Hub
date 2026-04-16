from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 32
    image_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    num_keypoints: int = 10
    noise_std: float = 0.03
    line_sigma: float = 1.2
    joint_sigma: float = 1.7


class SyntheticHandPoseDataset:
    """Render a compact grayscale hand skeleton with ten normalized keypoints."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects grayscale inputs.")
        if int(cfg.num_keypoints) != 10:
            raise ValueError("This lesson uses exactly 10 keypoints.")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if float(cfg.line_sigma) <= 0.0:
            raise ValueError("line_sigma must be > 0")
        if float(cfg.joint_sigma) <= 0.0:
            raise ValueError("joint_sigma must be > 0")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _segment_distance(
        self,
        xx: np.ndarray,
        yy: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> np.ndarray:
        vx = float(end[0] - start[0])
        vy = float(end[1] - start[1])
        denom = vx * vx + vy * vy
        if denom <= 1e-6:
            return np.sqrt((xx - float(start[0])) ** 2 + (yy - float(start[1])) ** 2)

        proj = ((xx - float(start[0])) * vx + (yy - float(start[1])) * vy) / denom
        proj = np.clip(proj, 0.0, 1.0)
        closest_x = float(start[0]) + proj * vx
        closest_y = float(start[1]) + proj * vy
        return np.sqrt((xx - closest_x) ** 2 + (yy - closest_y) ** 2)

    def _sample_keypoints(self, idx: int) -> np.ndarray:
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))
        size = int(self.cfg.image_size)

        center_x = float(rng.uniform(0.42, 0.58) * (size - 1))
        wrist_y = float(rng.uniform(0.66, 0.80) * (size - 1))
        spread = float(rng.uniform(0.08, 0.14) * size)
        palm_rise = float(rng.uniform(0.10, 0.16) * size)
        finger_len = float(rng.uniform(0.16, 0.24) * size)
        thumb_len = float(rng.uniform(0.14, 0.20) * size)
        tilt = float(rng.uniform(-0.08, 0.08) * size)

        wrist = np.array([center_x, wrist_y], dtype=np.float32)
        thumb_tip = wrist + np.array([-1.5 * spread, -0.35 * thumb_len + 0.35 * tilt], dtype=np.float32)
        index_base = wrist + np.array([-0.75 * spread, -0.45 * palm_rise], dtype=np.float32)
        index_tip = index_base + np.array([-0.18 * spread, -finger_len], dtype=np.float32)
        middle_base = wrist + np.array([-0.20 * spread, -0.62 * palm_rise], dtype=np.float32)
        middle_tip = middle_base + np.array([0.02 * spread, -1.10 * finger_len], dtype=np.float32)
        ring_base = wrist + np.array([0.35 * spread, -0.52 * palm_rise], dtype=np.float32)
        ring_tip = ring_base + np.array([0.12 * spread, -0.97 * finger_len], dtype=np.float32)
        pinky_base = wrist + np.array([0.86 * spread, -0.38 * palm_rise], dtype=np.float32)
        pinky_tip = pinky_base + np.array([0.28 * spread, -0.82 * finger_len], dtype=np.float32)

        keypoints = np.stack(
            [
                wrist,
                thumb_tip,
                index_base,
                index_tip,
                middle_base,
                middle_tip,
                ring_base,
                ring_tip,
                pinky_base,
                pinky_tip,
            ],
            axis=0,
        )
        margin = 2.0
        np.clip(keypoints, margin, float(size - 1) - margin, out=keypoints)
        return keypoints.astype(np.float32)

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        keypoints = self._sample_keypoints(int(idx))
        rng = np.random.default_rng(int(self.cfg.seed) * 97_409 + int(idx))

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        image = np.full((size, size), 0.07, dtype=np.float32)
        image += 0.04 * (1.0 - yy / max(float(size - 1), 1.0))

        segments = [
            (keypoints[0], keypoints[1], 0.20),
            (keypoints[0], keypoints[2], 0.18),
            (keypoints[2], keypoints[3], 0.22),
            (keypoints[0], keypoints[4], 0.18),
            (keypoints[4], keypoints[5], 0.24),
            (keypoints[0], keypoints[6], 0.18),
            (keypoints[6], keypoints[7], 0.22),
            (keypoints[0], keypoints[8], 0.17),
            (keypoints[8], keypoints[9], 0.20),
        ]
        for start, end, weight in segments:
            dist = self._segment_distance(xx, yy, start, end)
            stroke = np.exp(
                -(dist**2) / (2.0 * float(self.cfg.line_sigma) * float(self.cfg.line_sigma))
            ).astype(np.float32)
            image += weight * stroke

        for point in keypoints:
            sigma = float(self.cfg.joint_sigma)
            joint = np.exp(
                -((xx - float(point[0])) ** 2 + (yy - float(point[1])) ** 2)
                / (2.0 * sigma * sigma)
            ).astype(np.float32)
            image += 0.18 * joint

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        keypoint_tensor = torch.from_numpy((keypoints / float(size - 1)).reshape(-1))
        return image_tensor, keypoint_tensor.to(torch.float32)


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticHandPoseDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticHandPoseDataset", "get_dataloaders"]
