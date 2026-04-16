import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_panoptic_segmentation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_83_synthetic_panoptic_segmentation.data import (
        DataConfig,
        SyntheticPanopticSegmentationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_83_synthetic_panoptic_segmentation.model import (
        ModelConfig,
        PanopticSegmentationModel,
        panoptic_segmentation_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=4,
        image_size=48,
        val_fraction=0.2,
        seed=13,
        num_workers=0,
        num_thing_classes=3,
        num_stuff_classes=2,
        max_instances=2,
    )
    ds = SyntheticPanopticSegmentationDataset(cfg)
    image, target = ds[0]

    assert tuple(image.shape) == (3, 48, 48)
    assert set(target.keys()) == {"semantic_labels", "instance_masks", "instance_classes"}
    assert tuple(target["semantic_labels"].shape) == (48, 48)
    assert tuple(target["instance_masks"].shape) == (2, 48, 48)
    assert tuple(target["instance_classes"].shape) == (2,)
    assert image.dtype == torch.float32
    assert target["semantic_labels"].dtype == torch.long
    assert target["instance_masks"].dtype == torch.float32
    assert target["instance_classes"].dtype == torch.long
    assert int(target["semantic_labels"].min().item()) >= 0
    assert int(target["semantic_labels"].max().item()) < 5
    assert int(target["instance_classes"].min().item()) >= 0
    assert int(target["instance_classes"].max().item()) < 3

    train_loader, _ = get_dataloaders(cfg)
    batch_images, batch_targets = next(iter(train_loader))
    assert tuple(batch_images.shape) == (4, 3, 48, 48)
    assert tuple(batch_targets["semantic_labels"].shape) == (4, 48, 48)
    assert tuple(batch_targets["instance_masks"].shape) == (4, 2, 48, 48)
    assert tuple(batch_targets["instance_classes"].shape) == (4, 2)

    model = PanopticSegmentationModel(
        ModelConfig(
            in_channels=3,
            num_thing_classes=3,
            num_stuff_classes=2,
            max_instances=2,
            family="panoptic_fpn",
            variant="panoptic_fpn_tiny",
            width_mult=0.5,
        )
    )
    outputs = model(batch_images)
    assert set(outputs.keys()) == {
        "semantic_logits",
        "query_cls_logits",
        "query_boxes",
        "mask_logits",
        "panoptic_map",
    }
    assert tuple(outputs["semantic_logits"].shape) == (4, 5, 48, 48)
    assert tuple(outputs["query_cls_logits"].shape[:2]) == (4, 8)
    assert tuple(outputs["query_cls_logits"].shape[2:]) == (3,)
    assert tuple(outputs["mask_logits"].shape[:2]) == (4, 8)
    assert tuple(outputs["mask_logits"].shape[2:]) == (48, 48)
    assert tuple(outputs["panoptic_map"].shape) == (4, 48, 48)

    loss, parts = panoptic_segmentation_loss(outputs, batch_targets, max_instances=2)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"semantic_loss", "instance_cls_loss", "instance_mask_loss"}
    assert float(parts["semantic_loss"]) >= 0.0
    assert float(parts["instance_cls_loss"]) >= 0.0
    assert float(parts["instance_mask_loss"]) >= 0.0
    loss.backward()


def test_vision_panoptic_segmentation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_83_synthetic_panoptic_segmentation.data import DataConfig
    from tracks.vision.lesson_83_synthetic_panoptic_segmentation.model import ModelConfig
    from tracks.vision.lesson_83_synthetic_panoptic_segmentation.train import (
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
            run_name="pytest_panoptic_segmentation_smoke",
        ),
        DataConfig(
            num_samples=56,
            batch_size=4,
            image_size=48,
            val_fraction=0.2,
            seed=23,
            num_workers=0,
            num_thing_classes=3,
            num_stuff_classes=2,
            max_instances=2,
        ),
        ModelConfig(
            in_channels=3,
            num_thing_classes=3,
            num_stuff_classes=2,
            max_instances=2,
            family="panoptic_fpn",
            variant="panoptic_fpn_tiny",
            width_mult=0.5,
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_83_synthetic_panoptic_segmentation"
        / "pytest_panoptic_segmentation_smoke"
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
    row = metrics[0]
    for key in (
        "train_loss",
        "train_semantic_loss",
        "train_instance_cls_loss",
        "train_instance_mask_loss",
        "train_semantic_acc",
        "eval_loss",
        "eval_semantic_loss",
        "eval_instance_cls_loss",
        "eval_instance_mask_loss",
        "eval_semantic_acc",
    ):
        assert key in row
        assert float(row[key]) >= 0.0

