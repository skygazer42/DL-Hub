"""Small runnable method kits for cross-cutting DL-Hub topics.

These utilities intentionally stay dependency-light.  They provide concrete code
surfaces for methodology topics that cut across many model zoos, such as NAS,
AutoML, pruning, distillation, SLAM-style pose graphs, and metaverse scene
assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from statistics import mean


@dataclass(frozen=True)
class SearchCandidate:
    name: str
    width: int
    depth: int
    score: float


@dataclass(frozen=True)
class SearchResult:
    best: SearchCandidate
    ranked: tuple[SearchCandidate, ...]


def rank_nas_candidates(candidates: list[SearchCandidate]) -> SearchResult:
    """Rank NAS/AutoML candidates by score, then by compactness."""

    if not candidates:
        raise ValueError("candidates must not be empty")
    ranked = tuple(
        sorted(
            candidates,
            key=lambda item: (-float(item.score), int(item.width) * int(item.depth), item.name),
        )
    )
    return SearchResult(best=ranked[0], ranked=ranked)


def make_magnitude_pruning_mask(weights: list[float], *, keep_fraction: float) -> list[int]:
    """Return a deterministic magnitude-pruning mask for a flat weight list."""

    if not weights:
        raise ValueError("weights must not be empty")
    keep_fraction = float(keep_fraction)
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")

    keep = max(1, int(round(len(weights) * keep_fraction)))
    order = sorted(range(len(weights)), key=lambda idx: (abs(float(weights[idx])), idx), reverse=True)
    keep_idx = set(order[:keep])
    return [1 if idx in keep_idx else 0 for idx in range(len(weights))]


def distillation_temperature_loss(
    student_logits: list[float], teacher_logits: list[float], *, temperature: float = 2.0
) -> float:
    """Compute a tiny mean-squared distillation loss on temperature-scaled logits."""

    if len(student_logits) != len(teacher_logits):
        raise ValueError("student and teacher logits must have the same length")
    if not student_logits:
        raise ValueError("logits must not be empty")
    t = float(temperature)
    if t <= 0:
        raise ValueError("temperature must be positive")
    return mean(((float(s) / t) - (float(v) / t)) ** 2 for s, v in zip(student_logits, teacher_logits))


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float


def compose_pose(base: Pose2D, delta: Pose2D) -> Pose2D:
    """Compose a 2D SLAM odometry delta onto a base pose."""

    c = cos(float(base.theta))
    s = sin(float(base.theta))
    x = float(base.x) + c * float(delta.x) - s * float(delta.y)
    y = float(base.y) + s * float(delta.x) + c * float(delta.y)
    theta = float(base.theta) + float(delta.theta)
    return Pose2D(x=x, y=y, theta=theta)


@dataclass(frozen=True)
class SceneAsset:
    asset_id: str
    modality: str
    position: tuple[float, float, float]
    tags: tuple[str, ...] = ()


def summarize_scene_assets(assets: list[SceneAsset]) -> dict[str, object]:
    """Summarize a lightweight metaverse/3D-scene asset catalog."""

    by_modality: dict[str, int] = {}
    for asset in assets:
        by_modality[asset.modality] = by_modality.get(asset.modality, 0) + 1
    return {
        "count": len(assets),
        "modalities": dict(sorted(by_modality.items())),
        "tag_count": sum(len(asset.tags) for asset in assets),
    }


@dataclass(frozen=True)
class CapsuleRoutingState:
    logits: tuple[float, ...]
    couplings: tuple[float, ...]


def normalize_capsule_routing(logits: list[float]) -> CapsuleRoutingState:
    """Normalize routing logits into capsule coupling coefficients."""

    if not logits:
        raise ValueError("logits must not be empty")
    exps = [2.718281828459045 ** float(value) for value in logits]
    total = sum(exps)
    couplings = tuple(value / total for value in exps)
    return CapsuleRoutingState(logits=tuple(float(value) for value in logits), couplings=couplings)


def discounted_returns(rewards: list[float], *, gamma: float = 0.99) -> list[float]:
    """Compute reinforcement-learning discounted returns for one trajectory."""

    if not rewards:
        raise ValueError("rewards must not be empty")
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    out = [0.0 for _ in rewards]
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        out[idx] = running
    return out


def epsilon_greedy_action(values: list[float], *, epsilon: float, step: int = 0) -> int:
    """Pick a deterministic epsilon-greedy action for tiny RL examples.

    The exploratory branch is deterministic on `step` so tests and tutorials do
    not depend on random state.
    """

    if not values:
        raise ValueError("values must not be empty")
    epsilon = float(epsilon)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    explore_slot = int(round(1.0 / max(epsilon, 1e-12))) if epsilon > 0 else 0
    if epsilon > 0 and explore_slot > 0 and int(step) % explore_slot == 0:
        return int(step) % len(values)
    return max(range(len(values)), key=lambda idx: (float(values[idx]), -idx))


__all__ = [
    "CapsuleRoutingState",
    "Pose2D",
    "SceneAsset",
    "SearchCandidate",
    "SearchResult",
    "compose_pose",
    "discounted_returns",
    "distillation_temperature_loss",
    "epsilon_greedy_action",
    "make_magnitude_pruning_mask",
    "normalize_capsule_routing",
    "rank_nas_candidates",
    "summarize_scene_assets",
]
