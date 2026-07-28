import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_action_recognition_batch_shapes() -> None:
    from tracks.multimodal.lesson_34_video_text_action_recognition.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        num_frames=10,
        feature_dim=24,
        max_text_length=10,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["video_features"].shape == (8, 10, 24)
    assert batch["query_ids"].shape == (8, 10)
    assert batch["attention_mask"].shape == (8, 10)
    assert batch["label"].shape == (8,)
    assert len(batch["query_text"]) == 8
    assert len(batch["action_type"]) == 8
    assert "recognize" in vocab.token_to_id
    assert "action" in vocab.token_to_id
    assert "jump" in vocab.token_to_id
    assert "wave" in vocab.token_to_id
    assert "sit" in vocab.token_to_id


def test_multimodal_action_recognition_model_outputs() -> None:
    from tracks.multimodal.lesson_34_video_text_action_recognition.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_34_video_text_action_recognition.model import (
        ActionRecognitionModelConfig,
        CompactActionRecognitionModel,
        action_recognition_loss,
        classification_accuracy,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        num_frames=10,
        feature_dim=24,
        max_text_length=10,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = CompactActionRecognitionModel(
        ActionRecognitionModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_frames=data_cfg.num_frames,
            feature_dim=data_cfg.feature_dim,
            hidden_dim=40,
            text_dim=32,
            num_classes=3,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "pred_labels"}
    assert outputs["logits"].shape == (8, 3)
    assert outputs["pred_labels"].shape == (8,)

    losses = action_recognition_loss(
        logits=outputs["logits"],
        labels=batch["label"],
    )
    assert set(losses) >= {"loss", "cls_loss"}
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])

    acc = classification_accuracy(outputs["logits"], batch["label"])
    assert 0.0 <= acc <= 1.0


def test_multimodal_action_recognition_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_34_video_text_action_recognition"
        / "pytest_action_recognition_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_34_video_text_action_recognition.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--num-frames",
            "10",
            "--feature-dim",
            "24",
            "--max-text-length",
            "10",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_action_recognition_smoke",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
