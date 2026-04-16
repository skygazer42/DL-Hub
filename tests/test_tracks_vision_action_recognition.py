import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_action_recognition_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_87_synthetic_action_recognition.data import (
        DataConfig,
        SyntheticActionRecognitionDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_87_synthetic_action_recognition.model import (
        ActionRecognitionModel,
        ModelConfig,
        action_recognition_accuracy,
        action_recognition_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=4,
        seq_len=8,
        image_size=32,
        val_fraction=0.25,
        seed=7,
        num_workers=0,
        in_channels=1,
        num_classes=4,
        noise_std=0.01,
        motion_jitter=0.12,
    )
    ds = SyntheticActionRecognitionDataset(cfg)
    clip, target = ds[0]

    assert tuple(clip.shape) == (8, 1, 32, 32)
    assert set(target.keys()) == {"action_label"}
    assert tuple(target["action_label"].shape) == ()
    assert clip.dtype == torch.float32
    assert target["action_label"].dtype == torch.int64
    assert 0 <= int(target["action_label"].item()) < 4

    train_loader, _ = get_dataloaders(cfg)
    batch_clips, batch_targets = next(iter(train_loader))
    assert tuple(batch_clips.shape) == (4, 8, 1, 32, 32)
    assert tuple(batch_targets["action_label"].shape) == (4,)

    model = ActionRecognitionModel(
        ModelConfig(
            in_channels=1,
            num_classes=4,
            backbone_variant="c3d_tiny",
            backbone_width_mult=0.5,
            dropout=0.1,
        )
    )
    outputs = model(batch_clips)
    assert set(outputs.keys()) == {"action_logits"}
    assert tuple(outputs["action_logits"].shape) == (4, 4)

    loss, parts = action_recognition_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"action_loss"}
    assert float(parts["action_loss"]) >= 0.0
    assert 0.0 <= action_recognition_accuracy(outputs["action_logits"], batch_targets["action_label"]) <= 1.0
    loss.backward()


def test_vision_action_recognition_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_87_synthetic_action_recognition.data import DataConfig
    from tracks.vision.lesson_87_synthetic_action_recognition.model import ModelConfig
    from tracks.vision.lesson_87_synthetic_action_recognition.train import (
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
            run_name="pytest_action_recognition_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            seq_len=8,
            image_size=32,
            val_fraction=0.25,
            seed=11,
            num_workers=0,
            in_channels=1,
            num_classes=4,
            noise_std=0.01,
            motion_jitter=0.12,
        ),
        ModelConfig(
            in_channels=1,
            num_classes=4,
            backbone_variant="c3d_tiny",
            backbone_width_mult=0.5,
            dropout=0.1,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_87_synthetic_action_recognition" / "pytest_action_recognition_smoke"
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
    for key in ("train_loss", "train_action_loss", "train_acc", "eval_loss", "eval_action_loss", "eval_acc"):
        assert key in record
        assert float(record[key]) >= 0.0
