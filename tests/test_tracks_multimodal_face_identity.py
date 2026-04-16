import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_multimodal_face_identity_batch_shapes() -> None:
    from tracks.multimodal.lesson_37_face_identity_vlm_recognition.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=40,
        max_text_length=10,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 1, 40, 40)
    assert batch["identity_ids"].shape == (8, 10)
    assert batch["identity_mask"].shape == (8, 10)
    assert batch["labels"].shape == (8,)
    assert len(batch["identity_text"]) == 8
    assert len(batch["identity_name"]) == 8
    assert "identify" in vocab.token_to_id
    assert "face" in vocab.token_to_id
    assert "alice" in vocab.token_to_id


def test_multimodal_face_identity_model_outputs() -> None:
    from tracks.multimodal.lesson_37_face_identity_vlm_recognition.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_37_face_identity_vlm_recognition.model import (
        FaceIdentityConfig,
        ToyFaceIdentityVLM,
        face_identity_accuracy,
        face_identity_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=40,
        max_text_length=10,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyFaceIdentityVLM(
        FaceIdentityConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_identities=4,
            hidden_dim=48,
            text_dim=32,
            vision_width=24,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "probabilities", "pred_labels"}
    assert outputs["logits"].shape == (8, 4)
    assert outputs["probabilities"].shape == (8, 4)
    assert outputs["pred_labels"].shape == (8,)
    assert torch.allclose(
        outputs["probabilities"].sum(dim=1),
        torch.ones(8, dtype=outputs["probabilities"].dtype),
        atol=1e-5,
    )

    loss = face_identity_loss(outputs["logits"], batch["labels"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert 0.0 <= face_identity_accuracy(outputs["logits"], batch["labels"]) <= 1.0


def test_multimodal_face_identity_training_smoke(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.multimodal.lesson_37_face_identity_vlm_recognition.data import DataConfig
    from tracks.multimodal.lesson_37_face_identity_vlm_recognition.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path / "outputs"))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_face_identity_smoke",
            hidden_dim=48,
            text_dim=32,
            vision_width=24,
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            image_size=40,
            max_text_length=10,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
        ),
    )

    assert exit_code == 0
    run_dir = (
        tmp_path
        / "outputs"
        / "multimodal"
        / "lesson_37_face_identity_vlm_recognition"
        / "pytest_face_identity_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["epoch"] == 1


def test_multimodal_face_identity_dry_run() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_37_face_identity_vlm_recognition",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.multimodal.lesson_37_face_identity_vlm_recognition.train" in proc.stdout
