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
    num_keypoints: int = 8
    noise_std: float = 0.03
    line_sigma: float = 1.3
    joint_sigma: float = 1.8


class SyntheticHumanPoseDataset:
    """Render simple stick figures with eight normalized pose keypoints."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects grayscale inputs.")
        if int(cfg.num_keypoints) != 8:
            raise ValueError("This lesson uses exactly 8 keypoints.")
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

        body_scale = float(rng.uniform(0.90, 1.12))
        center_x = float(rng.uniform(0.40, 0.60) * (size - 1))
        shoulder_y = float(rng.uniform(0.28, 0.38) * (size - 1))
        lean = float(rng.uniform(-0.10, 0.10) * size)

        shoulder_span = float(rng.uniform(0.18, 0.26) * size * body_scale)
        torso_len = float(rng.uniform(0.18, 0.24) * size * body_scale)
        head_offset = float(rng.uniform(0.12, 0.16) * size * body_scale)
        arm_reach = float(rng.uniform(0.16, 0.26) * size * body_scale)
        hand_drop = float(rng.uniform(0.10, 0.24) * size * body_scale)
        left_hand_lift = float(rng.uniform(-0.10, 0.08) * size)
        right_hand_lift = float(rng.uniform(-0.10, 0.08) * size)
        stride = float(rng.uniform(0.08, 0.18) * size * body_scale)
        leg_len = float(rng.uniform(0.24, 0.32) * size * body_scale)
        left_leg_bend = float(rng.uniform(-0.04, 0.04) * size)
        right_leg_bend = float(rng.uniform(-0.04, 0.04) * size)

        shoulder_mid = np.array([center_x, shoulder_y], dtype=np.float32)
        head = np.array([center_x + 0.45 * lean, shoulder_y - head_offset], dtype=np.float32)
        left_shoulder = shoulder_mid + np.array([-0.5 * shoulder_span, 0.0], dtype=np.float32)
        right_shoulder = shoulder_mid + np.array([0.5 * shoulder_span, 0.0], dtype=np.float32)
        pelvis = shoulder_mid + np.array([lean, torso_len], dtype=np.float32)
        left_hand = left_shoulder + np.array(
            [-arm_reach, hand_drop + left_hand_lift],
            dtype=np.float32,
        )
        right_hand = right_shoulder + np.array(
            [arm_reach, hand_drop + right_hand_lift],
            dtype=np.float32,
        )
        left_foot = pelvis + np.array([-stride, leg_len + left_leg_bend], dtype=np.float32)
        right_foot = pelvis + np.array([stride, leg_len + right_leg_bend], dtype=np.float32)

        keypoints = np.stack(
            [
                head,
                left_shoulder,
                right_shoulder,
                left_hand,
                right_hand,
                pelvis,
                left_foot,
                right_foot,
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
        image += 0.03 * (1.0 - yy / max(float(size - 1), 1.0))

        shoulder_mid = 0.5 * (keypoints[1] + keypoints[2])
        segments = [
            (keypoints[0], shoulder_mid, 0.26),
            (keypoints[1], keypoints[2], 0.18),
            (shoulder_mid, keypoints[5], 0.24),
            (keypoints[1], keypoints[3], 0.22),
            (keypoints[2], keypoints[4], 0.22),
            (keypoints[5], keypoints[6], 0.22),
            (keypoints[5], keypoints[7], 0.22),
        ]
        for start, end, weight in segments:
            dist = self._segment_distance(xx, yy, start, end)
            stroke = np.exp(
                -(dist**2) / (2.0 * float(self.cfg.line_sigma) * float(self.cfg.line_sigma))
            ).astype(np.float32)
            image += weight * stroke

        for joint_idx, point in enumerate(keypoints):
            sigma = float(self.cfg.joint_sigma)
            if joint_idx == 0:
                sigma *= 1.35
            joint = np.exp(
                -((xx - float(point[0])) ** 2 + (yy - float(point[1])) ** 2)
                / (2.0 * sigma * sigma)
            ).astype(np.float32)
            image += 0.20 * joint

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        keypoint_tensor = torch.from_numpy((keypoints / float(size - 1)).reshape(-1))
        return image_tensor, keypoint_tensor.to(torch.float32)


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticHumanPoseDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticHumanPoseDataset", "get_dataloaders"]
