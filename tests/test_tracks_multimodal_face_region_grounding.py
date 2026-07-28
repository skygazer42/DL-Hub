import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_multimodal_face_region_grounding_batch_shapes() -> None:
    from tracks.multimodal.lesson_42_face_region_grounding_vlm.data import DataConfig, get_dataloaders

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
    assert batch["target_boxes"].shape == (8, 4)
    assert len(batch["region_name"]) == 8
    assert len(batch["query_text"]) == 8
    assert "locate" in vocab.token_to_id
    assert "face" in vocab.token_to_id
    assert "region" in vocab.token_to_id
    assert "eyes" in vocab.token_to_id
    assert "mouth" in vocab.token_to_id


def test_multimodal_face_region_grounding_model_outputs() -> None:
    from tracks.multimodal.lesson_42_face_region_grounding_vlm.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_42_face_region_grounding_vlm.model import (
        FaceRegionGroundingConfig,
        CompactFaceRegionGroundingModel,
        box_iou_xyxy,
        face_region_grounding_loss,
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

    model = CompactFaceRegionGroundingModel(
        FaceRegionGroundingConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            hidden_dim=64,
            text_dim=32,
            vision_width=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"pred_boxes"}
    assert outputs["pred_boxes"].shape == (8, 4)
    assert torch.all(outputs["pred_boxes"] >= 0.0)
    assert torch.all(outputs["pred_boxes"] <= 1.0)

    loss = face_region_grounding_loss(outputs["pred_boxes"], batch["target_boxes"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    iou = box_iou_xyxy(outputs["pred_boxes"], batch["target_boxes"])
    assert iou.shape == (8,)
    assert torch.all(iou >= 0.0)
    assert torch.all(iou <= 1.0)


def test_multimodal_face_region_grounding_training_smoke(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.multimodal.lesson_42_face_region_grounding_vlm.data import DataConfig
    from tracks.multimodal.lesson_42_face_region_grounding_vlm.train import TrainConfig, run_training

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
            run_name="pytest_face_region_grounding_smoke",
            hidden_dim=64,
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
        / "lesson_42_face_region_grounding_vlm"
        / "pytest_face_region_grounding_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["epoch"] == 1


def test_multimodal_face_region_grounding_dry_run() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_42_face_region_grounding_vlm",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.multimodal.lesson_42_face_region_grounding_vlm.train" in proc.stdout
