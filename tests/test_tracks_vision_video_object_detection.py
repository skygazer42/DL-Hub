import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_video_object_detection_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_66_synthetic_video_object_detection.data import (
        DataConfig,
        SyntheticVideoObjectDetectionDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_66_synthetic_video_object_detection.model import (
        ModelConfig,
        VideoObjectDetectionModel,
        video_object_detection_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=4,
        seq_len=5,
        image_size=32,
        val_fraction=0.25,
        seed=7,
        num_workers=0,
        in_channels=1,
        num_classes=3,
        max_objects=2,
        noise_std=0.01,
    )
    ds = SyntheticVideoObjectDetectionDataset(cfg)
    clip, target = ds[0]

    assert tuple(clip.shape) == (5, 1, 32, 32)
    assert set(target.keys()) == {"boxes", "labels", "present"}
    assert tuple(target["boxes"].shape) == (2, 4)
    assert tuple(target["labels"].shape) == (2,)
    assert tuple(target["present"].shape) == (2,)

    train_loader, _ = get_dataloaders(cfg)
    batch_clips, batch_targets = next(iter(train_loader))
    assert tuple(batch_clips.shape) == (4, 5, 1, 32, 32)
    assert tuple(batch_targets["boxes"].shape) == (4, 2, 4)
    assert tuple(batch_targets["labels"].shape) == (4, 2)
    assert tuple(batch_targets["present"].shape) == (4, 2)

    model = VideoObjectDetectionModel(
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
            max_objects=2,
            num_classes=3,
        )
    )
    outputs = model(batch_clips)
    assert set(outputs.keys()) == {"pred_boxes", "pred_scores", "pred_logits"}
    assert tuple(outputs["pred_boxes"].shape) == (4, 2, 4)
    assert tuple(outputs["pred_scores"].shape) == (4, 2)
    assert tuple(outputs["pred_logits"].shape) == (4, 2, 3)

    loss, parts = video_object_detection_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"box_loss", "score_loss", "class_loss"}
    assert float(parts["box_loss"]) >= 0.0
    assert float(parts["score_loss"]) >= 0.0
    assert float(parts["class_loss"]) >= 0.0
    loss.backward()


def test_vision_video_object_detection_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_66_synthetic_video_object_detection.data import DataConfig
    from tracks.vision.lesson_66_synthetic_video_object_detection.model import ModelConfig
    from tracks.vision.lesson_66_synthetic_video_object_detection.train import (
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
            run_name="pytest_video_object_detection_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            seq_len=5,
            image_size=32,
            val_fraction=0.25,
            seed=11,
            num_workers=0,
            in_channels=1,
            num_classes=3,
            max_objects=2,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
            max_objects=2,
            num_classes=3,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_66_synthetic_video_object_detection" / "pytest_video_object_detection_smoke"
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
    for key in ("train_loss", "train_box_loss", "train_score_loss", "train_class_loss", "eval_loss", "eval_box_loss", "eval_score_loss", "eval_class_loss"):
        assert key in record
        assert float(record[key]) >= 0.0
