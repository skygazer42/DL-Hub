import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_lane_topology_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_21_synthetic_lane_topology_estimation.data import (
        DataConfig,
        SyntheticLaneTopologyDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_21_synthetic_lane_topology_estimation.model import (
        LaneTopologyEstimator,
        ModelConfig,
        lane_topology_loss,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=4,
        image_size=48,
        val_fraction=0.25,
        seed=7,
        num_workers=0,
        min_lanes=2,
        max_lanes=4,
        lane_width=2.5,
        noise_std=0.01,
    )
    ds = SyntheticLaneTopologyDataset(cfg)
    image, target = ds[0]

    assert tuple(image.shape) == (1, 48, 48)
    assert set(target.keys()) == {"lane_heatmaps", "adjacency", "lane_presence"}
    assert tuple(target["lane_heatmaps"].shape) == (4, 48, 48)
    assert tuple(target["adjacency"].shape) == (4, 4)
    assert tuple(target["lane_presence"].shape) == (4,)
    assert image.dtype == torch.float32
    assert target["lane_heatmaps"].dtype == torch.float32
    assert target["adjacency"].dtype == torch.float32
    assert target["lane_presence"].dtype == torch.float32
    assert torch.allclose(target["adjacency"], target["adjacency"].transpose(0, 1))
    assert torch.allclose(torch.diag(target["adjacency"]), torch.zeros(4))
    assert 2 <= int(target["lane_presence"].sum().item()) <= 4

    train_loader, _ = get_dataloaders(cfg)
    batch_images, batch_targets = next(iter(train_loader))
    assert tuple(batch_images.shape) == (4, 1, 48, 48)
    assert tuple(batch_targets["lane_heatmaps"].shape) == (4, 4, 48, 48)

    model = LaneTopologyEstimator(
        ModelConfig(in_channels=1, hidden_channels=16, num_blocks=3, max_lanes=4)
    )
    outputs = model(batch_images)
    assert set(outputs.keys()) == {"lane_heatmaps", "adjacency_logits"}
    assert tuple(outputs["lane_heatmaps"].shape) == (4, 4, 48, 48)
    assert tuple(outputs["adjacency_logits"].shape) == (4, 4, 4)

    loss, parts = lane_topology_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"heatmap_loss", "adjacency_loss"}
    assert float(parts["heatmap_loss"]) >= 0.0
    assert float(parts["adjacency_loss"]) >= 0.0
    loss.backward()


def test_vision_lane_topology_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_21_synthetic_lane_topology_estimation.data import DataConfig
    from tracks.vision.lesson_21_synthetic_lane_topology_estimation.model import ModelConfig
    from tracks.vision.lesson_21_synthetic_lane_topology_estimation.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=3,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_lane_topology_smoke",
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            image_size=48,
            val_fraction=0.25,
            seed=11,
            num_workers=0,
            min_lanes=2,
            max_lanes=4,
            lane_width=2.5,
            noise_std=0.01,
        ),
        ModelConfig(in_channels=1, hidden_channels=16, num_blocks=3, max_lanes=4),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_21_synthetic_lane_topology_estimation" / "pytest_lane_topology_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "logs" / "train.log").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(metrics) == 1
    record = metrics[0]
    for key in (
        "train_loss",
        "train_heatmap_loss",
        "train_adjacency_loss",
        "eval_loss",
        "eval_heatmap_loss",
        "eval_adjacency_loss",
        "eval_edge_acc",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
