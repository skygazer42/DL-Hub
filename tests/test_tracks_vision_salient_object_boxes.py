import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_salient_object_boxes_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_30_synthetic_salient_object_detection_boxes.data import (
        DataConfig,
        SyntheticSalientObjectBoxesDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_30_synthetic_salient_object_detection_boxes.model import (
        ModelConfig,
        SalientObjectBoxesModel,
        box_iou,
        salient_box_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=1,
    )

    ds = SyntheticSalientObjectBoxesDataset(cfg)
    image, target_box = ds[0]
    assert tuple(image.shape) == (1, 32, 32)
    assert tuple(target_box.shape) == (4,)
    assert image.dtype == torch.float32
    assert target_box.dtype == torch.float32
    assert torch.all(target_box >= 0.0)
    assert torch.all(target_box <= 1.0)

    train_loader, _ = get_dataloaders(cfg)
    images, boxes = next(iter(train_loader))
    assert tuple(images.shape) == (4, 1, 32, 32)
    assert tuple(boxes.shape) == (4, 4)

    model = SalientObjectBoxesModel(
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3)
    )
    outputs = model(images)
    assert tuple(outputs.shape) == (4, 4)

    loss, parts = salient_box_loss(outputs, boxes)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"l1_loss", "iou_loss"}
    assert float(parts["l1_loss"]) >= 0.0
    assert float(parts["iou_loss"]) >= 0.0
    assert 0.0 <= box_iou(outputs.detach(), boxes) <= 1.0
    loss.backward()


def test_vision_salient_object_boxes_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_30_synthetic_salient_object_detection_boxes.data import DataConfig
    from tracks.vision.lesson_30_synthetic_salient_object_detection_boxes.model import ModelConfig
    from tracks.vision.lesson_30_synthetic_salient_object_detection_boxes.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_salient_object_boxes_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=5,
            num_workers=0,
            in_channels=1,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_30_synthetic_salient_object_detection_boxes"
        / "pytest_salient_object_boxes_smoke"
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
        "train_l1_loss",
        "train_iou_loss",
        "train_iou",
        "eval_loss",
        "eval_l1_loss",
        "eval_iou_loss",
        "eval_iou",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
