import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_text_detection_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_26_synthetic_text_detection.data import (
        DataConfig,
        SyntheticTextDetectionDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_26_synthetic_text_detection.model import (
        ModelConfig,
        TextDetectionModel,
        bbox_iou,
        text_detection_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
    )

    ds = SyntheticTextDetectionDataset(cfg)
    image, target = ds[0]
    assert tuple(image.shape) == (3, 32, 32)
    assert set(target.keys()) == {"bbox", "score"}
    assert tuple(target["bbox"].shape) == (4,)
    assert target["score"].ndim == 0
    assert image.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    images, targets = next(iter(train_loader))
    assert tuple(images.shape) == (4, 3, 32, 32)
    assert tuple(targets["bbox"].shape) == (4, 4)
    assert tuple(targets["score"].shape) == (4,)

    model = TextDetectionModel(ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3))
    outputs = model(images)
    assert set(outputs.keys()) == {"bbox", "score_logits"}
    assert tuple(outputs["bbox"].shape) == (4, 4)
    assert tuple(outputs["score_logits"].shape) == (4,)

    loss, parts = text_detection_loss(outputs, targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"bbox_loss", "score_loss"}
    assert float(parts["bbox_loss"]) >= 0.0
    assert float(parts["score_loss"]) >= 0.0

    iou = bbox_iou(outputs["bbox"].detach(), targets["bbox"])
    assert tuple(iou.shape) == (4,)
    assert torch.all(iou >= 0.0)
    loss.backward()


def test_vision_text_detection_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_26_synthetic_text_detection.data import DataConfig
    from tracks.vision.lesson_26_synthetic_text_detection.model import ModelConfig
    from tracks.vision.lesson_26_synthetic_text_detection.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_text_detection_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=11,
            num_workers=0,
            in_channels=3,
        ),
        ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_26_synthetic_text_detection" / "pytest_text_detection_smoke"
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
        "train_bbox_loss",
        "train_score_loss",
        "eval_loss",
        "eval_bbox_loss",
        "eval_score_loss",
        "eval_iou",
        "eval_score_acc",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
