import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_detection3d_batch_contract_and_loss_smoke() -> None:
    from tracks.pointcloud.lesson_27_compact_3d_object_detection.data import (
        DataConfig,
        SyntheticObjectDetection3DDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_27_compact_3d_object_detection.model import (
        ModelConfig,
        CompactDetector3D,
        detection3d_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        num_points=64,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        noise_points=20,
    )

    ds = SyntheticObjectDetection3DDataset(cfg)
    points, box, label = ds[0]
    assert tuple(points.shape) == (64, 3)
    assert tuple(box.shape) == (7,)
    assert tuple(label.shape) == ()
    assert points.dtype == torch.float32
    assert box.dtype == torch.float32
    assert label.dtype == torch.long
    assert int(label.item()) in (0, 1)
    assert float(box[3].item()) > 0.0
    assert float(box[4].item()) > 0.0
    assert float(box[5].item()) > 0.0

    train_loader, _ = get_dataloaders(cfg)
    points_batch, box_batch, label_batch = next(iter(train_loader))
    assert tuple(points_batch.shape) == (4, 64, 3)
    assert tuple(box_batch.shape) == (4, 7)
    assert tuple(label_batch.shape) == (4,)

    model = CompactDetector3D(ModelConfig(hidden_features=48, num_classes=2))
    outputs = model(points_batch)
    assert set(outputs.keys()) == {"boxes", "class_logits"}
    assert tuple(outputs["boxes"].shape) == (4, 7)
    assert tuple(outputs["class_logits"].shape) == (4, 2)

    loss, parts = detection3d_loss(outputs, box_batch, label_batch)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"box_l1", "box_yaw", "cls_ce"}
    assert float(parts["box_l1"]) >= 0.0
    assert float(parts["box_yaw"]) >= 0.0
    assert float(parts["cls_ce"]) >= 0.0
    loss.backward()


def test_pointcloud_detection3d_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_27_compact_3d_object_detection.data import DataConfig
    from tracks.pointcloud.lesson_27_compact_3d_object_detection.model import ModelConfig
    from tracks.pointcloud.lesson_27_compact_3d_object_detection.train import (
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
            run_name="pytest_detection3d_smoke",
        ),
        DataConfig(
            num_samples=48,
            num_points=64,
            batch_size=4,
            val_fraction=0.25,
            seed=5,
            num_workers=0,
            noise_points=20,
        ),
        ModelConfig(hidden_features=48, num_classes=2),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path / "pointcloud" / "lesson_27_compact_3d_object_detection" / "pytest_detection3d_smoke"
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
        "train_box_l1",
        "train_cls_ce",
        "eval_loss",
        "eval_box_l1",
        "eval_cls_ce",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
