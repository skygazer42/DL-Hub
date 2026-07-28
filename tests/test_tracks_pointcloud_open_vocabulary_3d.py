import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_open_vocabulary_3d_batch_contract_and_loss_smoke() -> None:
    from tracks.pointcloud.lesson_31_compact_open_vocabulary_3d.data import (
        DataConfig,
        SyntheticOpenVocabulary3DDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_31_compact_open_vocabulary_3d.model import (
        ModelConfig,
        CompactOpenVocabulary3DModel,
        mask_iou,
        open_vocabulary_3d_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        num_points=48,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        max_text_length=8,
    )

    ds = SyntheticOpenVocabulary3DDataset(cfg)
    points, query_ids, query_mask, class_label, point_mask = ds[0]
    assert tuple(points.shape) == (48, 3)
    assert tuple(query_ids.shape) == (8,)
    assert tuple(query_mask.shape) == (8,)
    assert tuple(class_label.shape) == ()
    assert tuple(point_mask.shape) == (48,)
    assert points.dtype == torch.float32
    assert query_ids.dtype == torch.long
    assert query_mask.dtype == torch.float32
    assert class_label.dtype == torch.long
    assert point_mask.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    points_b, query_ids_b, query_mask_b, class_b, point_mask_b = next(iter(train_loader))
    assert tuple(points_b.shape) == (4, 48, 3)
    assert tuple(query_ids_b.shape) == (4, 8)
    assert tuple(query_mask_b.shape) == (4, 8)
    assert tuple(class_b.shape) == (4,)
    assert tuple(point_mask_b.shape) == (4, 48)

    model = CompactOpenVocabulary3DModel(
        ModelConfig(
            vocab_size=ds.vocab_size,
            pad_id=ds.pad_id,
            text_dim=24,
            point_dim=48,
            hidden_dim=48,
            num_classes=3,
        )
    )
    outputs = model(points_b, query_ids_b, query_mask_b)
    assert set(outputs) == {"class_logits", "mask_logits"}
    assert tuple(outputs["class_logits"].shape) == (4, 3)
    assert tuple(outputs["mask_logits"].shape) == (4, 48)

    loss, parts = open_vocabulary_3d_loss(
        outputs["class_logits"],
        outputs["mask_logits"],
        class_b,
        point_mask_b,
    )
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"class_loss", "mask_loss"}
    assert float(parts["class_loss"]) >= 0.0
    assert float(parts["mask_loss"]) >= 0.0
    loss.backward()

    iou = mask_iou(outputs["mask_logits"], point_mask_b)
    assert 0.0 <= iou <= 1.0


def test_pointcloud_open_vocabulary_3d_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_31_compact_open_vocabulary_3d.data import DataConfig
    from tracks.pointcloud.lesson_31_compact_open_vocabulary_3d.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=7,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_open_vocabulary_3d_smoke",
            text_dim=24,
            point_dim=48,
            hidden_dim=48,
        ),
        DataConfig(
            num_samples=64,
            num_points=48,
            batch_size=8,
            val_fraction=0.25,
            seed=3,
            num_workers=0,
            max_text_length=8,
        ),
    )

    assert exit_code == 0
    run_dir = (
        tmp_path
        / "pointcloud"
        / "lesson_31_compact_open_vocabulary_3d"
        / "pytest_open_vocabulary_3d_smoke"
    )
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
        "train_class_acc",
        "train_mask_iou",
        "eval_loss",
        "eval_class_acc",
        "eval_mask_iou",
    ):
        assert key in record
    assert float(record["train_loss"]) >= 0.0
    assert float(record["eval_loss"]) >= 0.0
    assert 0.0 <= float(record["train_class_acc"]) <= 1.0
    assert 0.0 <= float(record["eval_class_acc"]) <= 1.0
    assert 0.0 <= float(record["train_mask_iou"]) <= 1.0
    assert 0.0 <= float(record["eval_mask_iou"]) <= 1.0


def test_pointcloud_open_vocabulary_3d_dry_run() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "pointcloud",
            "lesson_31_compact_open_vocabulary_3d",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.pointcloud.lesson_31_compact_open_vocabulary_3d.train" in proc.stdout
