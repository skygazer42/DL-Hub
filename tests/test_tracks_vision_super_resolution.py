from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_super_resolution_data_pair_smoke() -> None:
    from tracks.vision.lesson_17_synthetic_super_resolution.data import DataConfig, get_dataloaders

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            in_channels=3,
            upscale_factor=2,
        )
    )
    lr, hr = next(iter(train_loader))
    assert tuple(lr.shape) == (4, 3, 16, 16)
    assert tuple(hr.shape) == (4, 3, 32, 32)
    assert hr.shape[-1] == lr.shape[-1] * 2
    assert hr.shape[-2] == lr.shape[-2] * 2


def test_vision_super_resolution_lesson_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_17_synthetic_super_resolution.train import (
        DataConfig,
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    tc = TrainConfig(
        epochs=1,
        learning_rate=1e-3,
        seed=0,
        device="cpu",
        max_train_batches=1,
        max_eval_batches=1,
        run_name="smoke",
        arch="sr:srcnn_tiny",
        in_channels=3,
        upscale_factor=2,
    )
    dc = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        upscale_factor=2,
    )
    run_training(tc, dc)

    run_dir = tmp_path / "vision" / "lesson_17_synthetic_super_resolution" / "smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
