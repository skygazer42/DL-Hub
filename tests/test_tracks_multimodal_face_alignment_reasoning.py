import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_multimodal_face_alignment_reasoning_batch_shapes() -> None:
    from tracks.multimodal.lesson_45_face_alignment_vlm_reasoning.data import DataConfig, get_dataloaders

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
    assert batch["query_ids"].shape == (8, 12)
    assert batch["query_mask"].shape == (8, 12)
    assert batch["target_points"].shape == (8, 5, 2)
    assert len(batch["query_text"]) == 8
    assert "align" in vocab.token_to_id
    assert "face" in vocab.token_to_id
    assert "landmarks" in vocab.token_to_id
    assert "canonical" in vocab.token_to_id


def test_multimodal_face_alignment_reasoning_model_outputs() -> None:
    from tracks.multimodal.lesson_45_face_alignment_vlm_reasoning.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_45_face_alignment_vlm_reasoning.model import (
        FaceAlignmentReasoningConfig,
        ToyFaceAlignmentReasoningModel,
        face_alignment_loss,
        mean_alignment_l2,
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

    model = ToyFaceAlignmentReasoningModel(
        FaceAlignmentReasoningConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            hidden_dim=64,
            text_dim=32,
            vision_width=32,
            num_points=5,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"pred_points"}
    assert outputs["pred_points"].shape == (8, 5, 2)
    assert torch.all(outputs["pred_points"] >= 0.0)
    assert torch.all(outputs["pred_points"] <= 1.0)

    loss = face_alignment_loss(outputs["pred_points"], batch["target_points"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    alignment_l2 = mean_alignment_l2(outputs["pred_points"], batch["target_points"])
    assert alignment_l2.shape == (8,)
    assert torch.all(alignment_l2 >= 0.0)


def test_multimodal_face_alignment_reasoning_training_smoke(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.multimodal.lesson_45_face_alignment_vlm_reasoning.data import DataConfig
    from tracks.multimodal.lesson_45_face_alignment_vlm_reasoning.train import TrainConfig, run_training

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
            run_name="pytest_face_alignment_reasoning_smoke",
            hidden_dim=64,
            text_dim=32,
            vision_width=32,
            num_points=5,
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
        / "lesson_45_face_alignment_vlm_reasoning"
        / "pytest_face_alignment_reasoning_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["epoch"] == 1


def test_multimodal_face_alignment_reasoning_dry_run() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_45_face_alignment_vlm_reasoning",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.multimodal.lesson_45_face_alignment_vlm_reasoning.train" in proc.stdout
