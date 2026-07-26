import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_facial_expression_batch_shapes() -> None:
    from tracks.multimodal.lesson_35_face_expression_vlm_recognition.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=8,
        feature_dim=16,
        max_text_length=10,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["face_features"].shape == (8, 16)
    assert batch["prompt_ids"].shape == (8, 10)
    assert batch["prompt_mask"].shape == (8, 10)
    assert batch["label"].shape == (8,)
    assert batch["label"].dtype == torch.long
    assert len(batch["emotion_label"]) == 8
    assert len(batch["prompt_text"]) == 8
    assert "classify" in vocab.token_to_id
    assert "expression" in vocab.token_to_id
    assert "happy" in vocab.token_to_id
    assert "neutral" in vocab.token_to_id


def test_multimodal_facial_expression_model_outputs() -> None:
    from tracks.multimodal.lesson_35_face_expression_vlm_recognition.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_35_face_expression_vlm_recognition.model import (
        FacialExpressionModelConfig,
        ToyFacialExpressionVLM,
        classification_accuracy,
        expression_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        feature_dim=16,
        max_text_length=10,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyFacialExpressionVLM(
        FacialExpressionModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            feature_dim=data_cfg.feature_dim,
            hidden_dim=40,
            text_dim=24,
            num_classes=4,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "probs", "pred_labels"}
    assert outputs["logits"].shape == (8, 4)
    assert outputs["probs"].shape == (8, 4)
    assert outputs["pred_labels"].shape == (8,)
    assert torch.allclose(
        outputs["probs"].sum(dim=1),
        torch.ones(8, dtype=outputs["probs"].dtype),
        atol=1e-5,
    )

    losses = expression_loss(logits=outputs["logits"], labels=batch["label"])
    assert set(losses) >= {"loss", "cls_loss"}
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])

    acc = classification_accuracy(outputs["logits"], batch["label"])
    assert 0.0 <= acc <= 1.0


def test_multimodal_facial_expression_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_35_face_expression_vlm_recognition"
        / "pytest_facial_expression_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_35_face_expression_vlm_recognition.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--feature-dim",
            "16",
            "--max-text-length",
            "10",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_facial_expression_smoke",
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
