import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_video_frame_interpolation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_68_synthetic_video_frame_interpolation.data import (
        DataConfig,
        SyntheticVideoFrameInterpolationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_68_synthetic_video_frame_interpolation.model import (
        ModelConfig,
        VideoFrameInterpolationModel,
        frame_interpolation_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=9,
        num_workers=0,
        in_channels=3,
        motion_pixels=3,
        noise_std=0.01,
    )
    ds = SyntheticVideoFrameInterpolationDataset(cfg)
    endpoints, target = ds[0]

    assert tuple(endpoints.shape) == (2, 3, 32, 32)
    assert tuple(target.shape) == (3, 32, 32)
    assert endpoints.dtype == torch.float32
    assert target.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    batch_endpoints, batch_target = next(iter(train_loader))
    assert tuple(batch_endpoints.shape) == (4, 2, 3, 32, 32)
    assert tuple(batch_target.shape) == (4, 3, 32, 32)

    model = VideoFrameInterpolationModel(
        ModelConfig(
            in_channels=3,
            hidden_channels=24,
            num_blocks=3,
        )
    )
    outputs = model(batch_endpoints)
    assert set(outputs.keys()) == {"mid"}
    assert tuple(outputs["mid"].shape) == (4, 3, 32, 32)

    loss, parts = frame_interpolation_loss(outputs, batch_target)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"l1_loss"}
    assert float(parts["l1_loss"]) >= 0.0
    loss.backward()


def test_vision_video_frame_interpolation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_68_synthetic_video_frame_interpolation.data import DataConfig
    from tracks.vision.lesson_68_synthetic_video_frame_interpolation.model import ModelConfig
    from tracks.vision.lesson_68_synthetic_video_frame_interpolation.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=13,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_video_frame_interp_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=17,
            num_workers=0,
            in_channels=3,
            motion_pixels=3,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=3,
            hidden_channels=24,
            num_blocks=3,
        ),
    )
    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_68_synthetic_video_frame_interpolation"
        / "pytest_video_frame_interp_smoke"
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
    for key in ("train_loss", "train_l1_loss", "eval_loss", "eval_l1_loss", "eval_psnr"):
        assert key in record
        assert float(record[key]) >= 0.0
