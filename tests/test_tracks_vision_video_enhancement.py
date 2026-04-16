import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_video_enhancement_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_72_synthetic_video_enhancement.data import (
        DataConfig,
        SyntheticVideoEnhancementDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_72_synthetic_video_enhancement.model import (
        ModelConfig,
        VideoEnhancementModel,
        psnr_from_mse,
        video_enhancement_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=4,
        seq_len=6,
        image_size=32,
        val_fraction=0.25,
        seed=7,
        num_workers=0,
        in_channels=1,
        noise_std=0.08,
        blur_kernel_size=5,
    )
    ds = SyntheticVideoEnhancementDataset(cfg)
    degraded, target = ds[0]

    assert tuple(degraded.shape) == (6, 1, 32, 32)
    assert set(target.keys()) == {"clean"}
    assert tuple(target["clean"].shape) == (6, 1, 32, 32)
    assert degraded.dtype == torch.float32
    assert target["clean"].dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    batch_degraded, batch_targets = next(iter(train_loader))
    assert tuple(batch_degraded.shape) == (4, 6, 1, 32, 32)
    assert tuple(batch_targets["clean"].shape) == (4, 6, 1, 32, 32)

    model = VideoEnhancementModel(
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
        )
    )
    outputs = model(batch_degraded)
    assert set(outputs.keys()) == {"enhanced"}
    assert tuple(outputs["enhanced"].shape) == (4, 6, 1, 32, 32)

    loss, parts = video_enhancement_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"reconstruction_loss"}
    assert float(parts["reconstruction_loss"]) >= 0.0
    assert psnr_from_mse(float(parts["reconstruction_loss"])) >= 0.0
    loss.backward()


def test_vision_video_enhancement_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_72_synthetic_video_enhancement.data import DataConfig
    from tracks.vision.lesson_72_synthetic_video_enhancement.model import ModelConfig
    from tracks.vision.lesson_72_synthetic_video_enhancement.train import (
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
            run_name="pytest_video_enhancement_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            seq_len=6,
            image_size=32,
            val_fraction=0.25,
            seed=11,
            num_workers=0,
            in_channels=1,
            noise_std=0.08,
            blur_kernel_size=5,
        ),
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_72_synthetic_video_enhancement" / "pytest_video_enhancement_smoke"
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
    for key in ("train_loss", "train_reconstruction_loss", "train_psnr", "eval_loss", "eval_reconstruction_loss", "eval_psnr"):
        assert key in record
        assert float(record[key]) >= 0.0
