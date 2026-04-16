import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_interactive_segmentation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_31_synthetic_interactive_segmentation.data import (
        DataConfig,
        SyntheticInteractiveSegmentationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_31_synthetic_interactive_segmentation.model import (
        InteractiveSegmentationModel,
        ModelConfig,
        interactive_segmentation_loss,
        mask_iou,
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

    ds = SyntheticInteractiveSegmentationDataset(cfg)
    image, click_map, target = ds[0]
    assert tuple(image.shape) == (1, 32, 32)
    assert tuple(click_map.shape) == (1, 32, 32)
    assert tuple(target.shape) == (1, 32, 32)
    assert image.dtype == torch.float32
    assert click_map.dtype == torch.float32
    assert target.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    images, click_maps, targets = next(iter(train_loader))
    assert tuple(images.shape) == (4, 1, 32, 32)
    assert tuple(click_maps.shape) == (4, 1, 32, 32)
    assert tuple(targets.shape) == (4, 1, 32, 32)

    model = InteractiveSegmentationModel(
        ModelConfig(in_channels=2, hidden_channels=24, num_blocks=3)
    )
    outputs = model(images, click_maps)
    assert tuple(outputs.shape) == (4, 1, 32, 32)

    loss, parts = interactive_segmentation_loss(outputs, targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"bce_loss", "dice_loss"}
    assert float(parts["bce_loss"]) >= 0.0
    assert float(parts["dice_loss"]) >= 0.0
    assert 0.0 <= mask_iou(outputs.detach(), targets) <= 1.0
    loss.backward()


def test_vision_interactive_segmentation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_31_synthetic_interactive_segmentation.data import DataConfig
    from tracks.vision.lesson_31_synthetic_interactive_segmentation.model import ModelConfig
    from tracks.vision.lesson_31_synthetic_interactive_segmentation.train import (
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
            run_name="pytest_interactive_segmentation_smoke",
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
        ModelConfig(in_channels=2, hidden_channels=24, num_blocks=3),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_31_synthetic_interactive_segmentation"
        / "pytest_interactive_segmentation_smoke"
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
