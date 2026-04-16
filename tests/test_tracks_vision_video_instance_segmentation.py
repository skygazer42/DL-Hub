import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_video_instance_segmentation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_74_synthetic_video_instance_segmentation.data import (
        DataConfig,
        SyntheticVideoInstanceSegmentationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_74_synthetic_video_instance_segmentation.model import (
        ModelConfig,
        VideoInstanceSegmentationModel,
        video_instance_segmentation_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=4,
        seq_len=5,
        image_size=32,
        val_fraction=0.25,
        seed=7,
        num_workers=0,
        in_channels=1,
        num_instances=3,
        noise_std=0.01,
    )
    ds = SyntheticVideoInstanceSegmentationDataset(cfg)
    clip, target = ds[0]

    assert tuple(clip.shape) == (5, 1, 32, 32)
    assert set(target.keys()) == {"instance_masks"}
    assert tuple(target["instance_masks"].shape) == (5, 3, 32, 32)
    assert clip.dtype == torch.float32
    assert target["instance_masks"].dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    batch_clips, batch_targets = next(iter(train_loader))
    assert tuple(batch_clips.shape) == (4, 5, 1, 32, 32)
    assert tuple(batch_targets["instance_masks"].shape) == (4, 5, 3, 32, 32)

    model = VideoInstanceSegmentationModel(
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
            num_instances=3,
        )
    )
    outputs = model(batch_clips)
    assert set(outputs.keys()) == {"instance_logits"}
    assert tuple(outputs["instance_logits"].shape) == (4, 5, 3, 32, 32)

    loss, parts = video_instance_segmentation_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"mask_bce_loss"}
    assert float(parts["mask_bce_loss"]) >= 0.0
    loss.backward()


def test_vision_video_instance_segmentation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_74_synthetic_video_instance_segmentation.data import DataConfig
    from tracks.vision.lesson_74_synthetic_video_instance_segmentation.model import ModelConfig
    from tracks.vision.lesson_74_synthetic_video_instance_segmentation.train import (
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
            run_name="pytest_video_instance_segmentation_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            seq_len=5,
            image_size=32,
            val_fraction=0.25,
            seed=11,
            num_workers=0,
            in_channels=1,
            num_instances=3,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
            num_instances=3,
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path / "vision" / "lesson_74_synthetic_video_instance_segmentation" / "pytest_video_instance_segmentation_smoke"
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
    for key in ("train_loss", "train_mask_bce_loss", "eval_loss", "eval_mask_bce_loss"):
        assert key in record
        assert float(record[key]) >= 0.0
