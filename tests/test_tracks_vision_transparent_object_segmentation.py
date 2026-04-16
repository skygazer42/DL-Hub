import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_transparent_object_segmentation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_79_synthetic_transparent_object_segmentation.data import (
        DataConfig,
        SyntheticTransparentObjectSegmentationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_79_synthetic_transparent_object_segmentation.model import (
        ModelConfig,
        TransparentObjectSegmentationModel,
        transparent_segmentation_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        alpha_min=0.25,
        alpha_max=0.75,
    )
    ds = SyntheticTransparentObjectSegmentationDataset(cfg)
    image, targets = ds[0]

    assert tuple(image.shape) == (3, 32, 32)
    assert set(targets.keys()) == {"mask", "alpha", "boundary"}
    assert tuple(targets["mask"].shape) == (1, 32, 32)
    assert tuple(targets["alpha"].shape) == (1, 32, 32)
    assert tuple(targets["boundary"].shape) == (1, 32, 32)
    assert image.dtype == torch.float32
    assert targets["mask"].dtype == torch.float32
    assert targets["alpha"].dtype == torch.float32
    assert targets["boundary"].dtype == torch.float32
    assert 0.0 <= float(image.min().item()) <= float(image.max().item()) <= 1.0
    assert 0.0 <= float(targets["mask"].min().item()) <= float(targets["mask"].max().item()) <= 1.0
    assert 0.0 <= float(targets["alpha"].min().item()) <= float(targets["alpha"].max().item()) <= 1.0

    train_loader, _ = get_dataloaders(cfg)
    batch_images, batch_targets = next(iter(train_loader))
    assert tuple(batch_images.shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["mask"].shape) == (4, 1, 32, 32)
    assert tuple(batch_targets["alpha"].shape) == (4, 1, 32, 32)
    assert tuple(batch_targets["boundary"].shape) == (4, 1, 32, 32)

    model = TransparentObjectSegmentationModel(
        ModelConfig(
            in_channels=3,
            arch="glassseg_toy",
            variant="glassseg_toy_tiny",
            width_mult=0.75,
        )
    )
    outputs = model(batch_images)
    assert set(outputs.keys()) == {"logits", "mask", "alpha", "boundary", "composite"}
    assert tuple(outputs["logits"].shape) == (4, 1, 32, 32)
    assert tuple(outputs["mask"].shape) == (4, 1, 32, 32)
    assert tuple(outputs["alpha"].shape) == (4, 1, 32, 32)
    assert tuple(outputs["boundary"].shape) == (4, 1, 32, 32)
    assert tuple(outputs["composite"].shape) == (4, 3, 32, 32)

    loss, parts = transparent_segmentation_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"mask_bce", "alpha_l1", "boundary_l1"}
    assert float(parts["mask_bce"]) >= 0.0
    assert float(parts["alpha_l1"]) >= 0.0
    assert float(parts["boundary_l1"]) >= 0.0
    loss.backward()


def test_vision_transparent_object_segmentation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_79_synthetic_transparent_object_segmentation.data import DataConfig
    from tracks.vision.lesson_79_synthetic_transparent_object_segmentation.model import ModelConfig
    from tracks.vision.lesson_79_synthetic_transparent_object_segmentation.train import (
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
            run_name="pytest_transparent_seg_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=11,
            num_workers=0,
            in_channels=3,
            alpha_min=0.25,
            alpha_max=0.75,
        ),
        ModelConfig(
            in_channels=3,
            arch="glassseg_toy",
            variant="glassseg_toy_tiny",
            width_mult=0.75,
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_79_synthetic_transparent_object_segmentation"
        / "pytest_transparent_seg_smoke"
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
        "train_mask_bce",
        "train_alpha_l1",
        "train_boundary_l1",
        "train_mask_iou",
        "eval_loss",
        "eval_mask_bce",
        "eval_alpha_l1",
        "eval_boundary_l1",
        "eval_mask_iou",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
