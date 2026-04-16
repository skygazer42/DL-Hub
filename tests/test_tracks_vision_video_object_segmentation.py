import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_video_object_segmentation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_73_synthetic_video_object_segmentation.data import (
        DataConfig,
        SyntheticVideoObjectSegmentationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_73_synthetic_video_object_segmentation.model import (
        ModelConfig,
        VideoObjectSegmentationModel,
        mask_iou,
        video_object_segmentation_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        seq_len=5,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=1,
        noise_std=0.04,
    )
    ds = SyntheticVideoObjectSegmentationDataset(cfg)
    video, target_mask = ds[0]
    assert tuple(video.shape) == (5, 1, 32, 32)
    assert tuple(target_mask.shape) == (5, 1, 32, 32)
    assert video.dtype == torch.float32
    assert target_mask.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    videos, masks = next(iter(train_loader))
    assert tuple(videos.shape) == (4, 5, 1, 32, 32)
    assert tuple(masks.shape) == (4, 5, 1, 32, 32)

    model = VideoObjectSegmentationModel(
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3)
    )
    logits = model(videos)
    assert tuple(logits.shape) == (4, 5, 1, 32, 32)

    loss, parts = video_object_segmentation_loss(logits, masks)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"bce_loss", "dice_loss"}
    assert float(parts["bce_loss"]) >= 0.0
    assert float(parts["dice_loss"]) >= 0.0
    assert 0.0 <= mask_iou(logits.detach(), masks) <= 1.0
    loss.backward()


def test_vision_video_object_segmentation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_73_synthetic_video_object_segmentation.data import DataConfig
    from tracks.vision.lesson_73_synthetic_video_object_segmentation.model import ModelConfig
    from tracks.vision.lesson_73_synthetic_video_object_segmentation.train import (
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
            run_name="pytest_video_object_segmentation_smoke",
        ),
        DataConfig(
            num_samples=40,
            batch_size=4,
            seq_len=5,
            image_size=32,
            val_fraction=0.2,
            seed=3,
            num_workers=0,
            in_channels=1,
            noise_std=0.04,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3),
    )
    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_73_synthetic_video_object_segmentation"
        / "pytest_video_object_segmentation_smoke"
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
        "train_bce_loss",
        "train_dice_loss",
        "train_iou",
        "eval_loss",
        "eval_bce_loss",
        "eval_dice_loss",
        "eval_iou",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
