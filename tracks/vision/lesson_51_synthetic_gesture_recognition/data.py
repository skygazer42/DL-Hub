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
    num_classes: int = 4
    noise_std: float = 0.025
    line_sigma: float = 1.4
    joint_sigma: float = 1.7


class SyntheticGestureRecognitionDataset:
    """Render simple stick figures with deterministic gesture classes."""

    gesture_names = ("rest", "left_wave", "right_wave", "hands_up")

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects grayscale inputs.")
        if int(cfg.num_classes) != 4:
            raise ValueError("This lesson uses exactly 4 gesture classes.")
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

    def _sample_pose(self, idx: int) -> tuple[np.ndarray, int]:
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))
        size = int(self.cfg.image_size)
        label = int((int(idx) + int(self.cfg.seed)) % int(self.cfg.num_classes))

        center_x = float(rng.uniform(0.42, 0.58) * (size - 1))
        shoulder_y = float(rng.uniform(0.28, 0.34) * (size - 1))
        shoulder_span = float(rng.uniform(0.20, 0.26) * size)
        torso_len = float(rng.uniform(0.19, 0.24) * size)
        head_offset = float(rng.uniform(0.10, 0.14) * size)
        arm_reach = float(rng.uniform(0.15, 0.22) * size)
        lower_arm = float(rng.uniform(0.12, 0.17) * size)
        hip_span = float(rng.uniform(0.11, 0.16) * size)
        leg_len = float(rng.uniform(0.24, 0.30) * size)

        head = np.array([center_x, shoulder_y - head_offset], dtype=np.float32)
        left_shoulder = np.array([center_x - 0.5 * shoulder_span, shoulder_y], dtype=np.float32)
        right_shoulder = np.array([center_x + 0.5 * shoulder_span, shoulder_y], dtype=np.float32)
        pelvis = np.array([center_x, shoulder_y + torso_len], dtype=np.float32)
        left_foot = pelvis + np.array([-hip_span, leg_len], dtype=np.float32)
        right_foot = pelvis + np.array([hip_span, leg_len], dtype=np.float32)

        left_elbow = left_shoulder + np.array([-0.55 * arm_reach, 0.38 * arm_reach], dtype=np.float32)
        right_elbow = right_shoulder + np.array([0.55 * arm_reach, 0.38 * arm_reach], dtype=np.float32)
        left_hand = left_elbow + np.array([-0.60 * lower_arm, 0.82 * lower_arm], dtype=np.float32)
        right_hand = right_elbow + np.array([0.60 * lower_arm, 0.82 * lower_arm], dtype=np.float32)

        if label == 1:
            left_elbow = left_shoulder + np.array([-0.35 * arm_reach, -0.65 * arm_reach], dtype=np.float32)
            left_hand = left_elbow + np.array([-0.95 * lower_arm, -0.25 * lower_arm], dtype=np.float32)
        elif label == 2:
            right_elbow = right_shoulder + np.array([0.35 * arm_reach, -0.65 * arm_reach], dtype=np.float32)
            right_hand = right_elbow + np.array([0.95 * lower_arm, -0.25 * lower_arm], dtype=np.float32)
        elif label == 3:
            left_elbow = left_shoulder + np.array([-0.22 * arm_reach, -0.82 * arm_reach], dtype=np.float32)
            right_elbow = right_shoulder + np.array([0.22 * arm_reach, -0.82 * arm_reach], dtype=np.float32)
            left_hand = left_elbow + np.array([-0.25 * lower_arm, -1.00 * lower_arm], dtype=np.float32)
            right_hand = right_elbow + np.array([0.25 * lower_arm, -1.00 * lower_arm], dtype=np.float32)

        keypoints = np.stack(
            [
                head,
                left_shoulder,
                right_shoulder,
                left_elbow,
                right_elbow,
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
        return keypoints.astype(np.float32), label

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        keypoints, label = self._sample_pose(int(idx))
        rng = np.random.default_rng(int(self.cfg.seed) * 97_409 + int(idx))

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        image = np.full((size, size), 0.06, dtype=np.float32)
        image += 0.03 * (1.0 - yy / max(float(size - 1), 1.0))

        shoulder_mid = 0.5 * (keypoints[1] + keypoints[2])
        segments = [
            (keypoints[0], shoulder_mid, 0.24),
            (keypoints[1], keypoints[2], 0.16),
            (shoulder_mid, keypoints[7], 0.24),
            (keypoints[1], keypoints[3], 0.20),
            (keypoints[3], keypoints[5], 0.22),
            (keypoints[2], keypoints[4], 0.20),
            (keypoints[4], keypoints[6], 0.22),
            (keypoints[7], keypoints[8], 0.20),
            (keypoints[7], keypoints[9], 0.20),
        ]
        for start, end, weight in segments:
            dist = self._segment_distance(xx, yy, start, end)
            stroke = np.exp(
                -(dist**2) / (2.0 * float(self.cfg.line_sigma) * float(self.cfg.line_sigma))
            ).astype(np.float32)
            image += weight * stroke

        for point_idx, point in enumerate(keypoints):
            sigma = float(self.cfg.joint_sigma)
            if point_idx in (0, 5, 6):
                sigma *= 1.1
            joint = np.exp(
                -((xx - float(point[0])) ** 2 + (yy - float(point[1])) ** 2)
                / (2.0 * sigma * sigma)
            ).astype(np.float32)
            image += 0.16 * joint

        # Add a gesture-dependent torso highlight so the classes remain easy to separate.
        if label == 1:
            image += 0.04 * np.clip((keypoints[1][0] - xx) / max(size * 0.25, 1.0), 0.0, 1.0)
        elif label == 2:
            image += 0.04 * np.clip((xx - keypoints[2][0]) / max(size * 0.25, 1.0), 0.0, 1.0)
        elif label == 3:
            image += 0.05 * np.exp(-((yy - keypoints[0][1]) ** 2) / (2.0 * 2.0 * 2.0)).astype(np.float32)

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return torch.from_numpy(image).unsqueeze(0), label


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticGestureRecognitionDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    def _collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
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
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticGestureRecognitionDataset", "get_dataloaders"]
