import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_road_scene_understanding_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_22_synthetic_road_scene_understanding.data import (
        DataConfig,
        SyntheticRoadSceneDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_22_synthetic_road_scene_understanding.model import (
        ModelConfig,
        RoadSceneUnderstandingModel,
        road_scene_loss,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=4,
        image_size=48,
        val_fraction=0.25,
        seed=7,
        num_workers=0,
        num_lane_slots=3,
        num_object_types=3,
        noise_std=0.01,
    )
    ds = SyntheticRoadSceneDataset(cfg)
    image, target = ds[0]

    assert tuple(image.shape) == (1, 48, 48)
    assert set(target.keys()) == {"lane_targets", "object_targets", "scene_label"}
    assert tuple(target["lane_targets"].shape) == (3,)
    assert tuple(target["object_targets"].shape) == (3,)
    assert tuple(target["scene_label"].shape) == ()
    assert image.dtype == torch.float32
    assert target["lane_targets"].dtype == torch.float32
    assert target["object_targets"].dtype == torch.float32
    assert target["scene_label"].dtype == torch.int64
    assert 0.0 <= float(target["lane_targets"].min().item()) <= float(target["lane_targets"].max().item()) <= 1.0
    assert 0.0 <= float(target["object_targets"].min().item()) <= float(target["object_targets"].max().item()) <= 1.0
    assert 0 <= int(target["scene_label"].item()) < 4

    train_loader, _ = get_dataloaders(cfg)
    batch_images, batch_targets = next(iter(train_loader))
    assert tuple(batch_images.shape) == (4, 1, 48, 48)
    assert tuple(batch_targets["lane_targets"].shape) == (4, 3)
    assert tuple(batch_targets["object_targets"].shape) == (4, 3)
    assert tuple(batch_targets["scene_label"].shape) == (4,)

    model = RoadSceneUnderstandingModel(
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
            num_lane_slots=3,
            num_object_types=3,
            num_scene_classes=4,
        )
    )
    outputs = model(batch_images)
    assert set(outputs.keys()) == {"lane_logits", "object_logits", "scene_logits"}
    assert tuple(outputs["lane_logits"].shape) == (4, 3)
    assert tuple(outputs["object_logits"].shape) == (4, 3)
    assert tuple(outputs["scene_logits"].shape) == (4, 4)

    loss, parts = road_scene_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"lane_loss", "object_loss", "scene_loss"}
    assert float(parts["lane_loss"]) >= 0.0
    assert float(parts["object_loss"]) >= 0.0
    assert float(parts["scene_loss"]) >= 0.0
    loss.backward()


def test_vision_road_scene_understanding_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_22_synthetic_road_scene_understanding.data import DataConfig
    from tracks.vision.lesson_22_synthetic_road_scene_understanding.model import ModelConfig
    from tracks.vision.lesson_22_synthetic_road_scene_understanding.train import (
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
            run_name="pytest_road_scene_smoke",
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            image_size=48,
            val_fraction=0.25,
            seed=11,
            num_workers=0,
            num_lane_slots=3,
            num_object_types=3,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
            num_lane_slots=3,
            num_object_types=3,
            num_scene_classes=4,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_22_synthetic_road_scene_understanding" / "pytest_road_scene_smoke"
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
        "train_lane_loss",
        "train_object_loss",
        "train_scene_loss",
        "eval_loss",
        "eval_lane_loss",
        "eval_object_loss",
        "eval_scene_loss",
        "eval_scene_acc",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
