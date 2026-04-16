import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_multimodal_face_caption_batch_shapes() -> None:
    from tracks.multimodal.lesson_40_face_caption_vlm_grounding.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=48,
        max_text_length=12,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 1, 48, 48)
    assert batch["caption_ids"].shape == (8, 12)
    assert batch["caption_mask"].shape == (8, 12)
    assert batch["labels"].shape == (8,)
    assert len(batch["caption_text"]) == 8
    assert "face" in vocab.token_to_id
    assert "caption" in vocab.token_to_id
    assert "match" in vocab.token_to_id
    assert "mismatch" in vocab.token_to_id


def test_multimodal_face_caption_model_outputs() -> None:
    from tracks.multimodal.lesson_40_face_caption_vlm_grounding.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_40_face_caption_vlm_grounding.model import (
        FaceCaptionGroundingConfig,
        ToyFaceCaptionGroundingModel,
        grounding_accuracy,
        grounding_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=48,
        max_text_length=12,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyFaceCaptionGroundingModel(
        FaceCaptionGroundingConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_classes=2,
            hidden_dim=48,
            text_dim=32,
            vision_width=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "probabilities", "pred_labels"}
    assert outputs["logits"].shape == (8, 2)
    assert outputs["probabilities"].shape == (8, 2)
    assert outputs["pred_labels"].shape == (8,)
    assert torch.allclose(
        outputs["probabilities"].sum(dim=1),
        torch.ones(8, dtype=outputs["probabilities"].dtype),
        atol=1e-5,
    )

    loss = grounding_loss(outputs["logits"], batch["labels"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert 0.0 <= grounding_accuracy(outputs["logits"], batch["labels"]) <= 1.0


def test_multimodal_face_caption_training_smoke(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.multimodal.lesson_40_face_caption_vlm_grounding.data import DataConfig
    from tracks.multimodal.lesson_40_face_caption_vlm_grounding.train import TrainConfig, run_training

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
            run_name="pytest_face_caption_smoke",
            hidden_dim=48,
            text_dim=32,
            vision_width=32,
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            image_size=48,
            max_text_length=12,
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
        / "lesson_40_face_caption_vlm_grounding"
        / "pytest_face_caption_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["epoch"] == 1


def test_multimodal_face_caption_dry_run() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_40_face_caption_vlm_grounding",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.multimodal.lesson_40_face_caption_vlm_grounding.train" in proc.stdout
