import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_video_understanding_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_70_synthetic_video_understanding.data import (
        DataConfig,
        SyntheticVideoUnderstandingDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_70_synthetic_video_understanding.model import (
        ModelConfig,
        VideoUnderstandingModel,
        video_understanding_accuracy,
        video_understanding_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=4,
        seq_len=6,
        image_size=32,
        val_fraction=0.25,
        seed=7,
        num_workers=0,
        in_channels=1,
        num_classes=4,
        noise_std=0.01,
    )
    ds = SyntheticVideoUnderstandingDataset(cfg)
    clip, target = ds[0]

    assert tuple(clip.shape) == (6, 1, 32, 32)
    assert set(target.keys()) == {"event_label"}
    assert tuple(target["event_label"].shape) == ()
    assert clip.dtype == torch.float32
    assert target["event_label"].dtype == torch.int64
    assert 0 <= int(target["event_label"].item()) < 4

    train_loader, _ = get_dataloaders(cfg)
    batch_clips, batch_targets = next(iter(train_loader))
    assert tuple(batch_clips.shape) == (4, 6, 1, 32, 32)
    assert tuple(batch_targets["event_label"].shape) == (4,)

    model = VideoUnderstandingModel(
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
            num_classes=4,
        )
    )
    outputs = model(batch_clips)
    assert set(outputs.keys()) == {"event_logits"}
    assert tuple(outputs["event_logits"].shape) == (4, 4)

    loss, parts = video_understanding_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"event_loss"}
    assert float(parts["event_loss"]) >= 0.0
    assert 0.0 <= video_understanding_accuracy(outputs["event_logits"], batch_targets["event_label"]) <= 1.0
    loss.backward()


def test_vision_video_understanding_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_70_synthetic_video_understanding.data import DataConfig
    from tracks.vision.lesson_70_synthetic_video_understanding.model import ModelConfig
    from tracks.vision.lesson_70_synthetic_video_understanding.train import (
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
            run_name="pytest_video_understanding_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            seq_len=6,
            image_size=32,
            val_fraction=0.25,
            seed=11,
            num_workers=0,
            in_channels=1,
            num_classes=4,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
            num_classes=4,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_70_synthetic_video_understanding" / "pytest_video_understanding_smoke"
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
    for key in ("train_loss", "train_event_loss", "train_acc", "eval_loss", "eval_event_loss", "eval_acc"):
        assert key in record
        assert float(record[key]) >= 0.0
