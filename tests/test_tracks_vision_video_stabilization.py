import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_video_stabilization_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_67_synthetic_video_stabilization.data import (
        DataConfig,
        SyntheticVideoStabilizationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_67_synthetic_video_stabilization.model import (
        ModelConfig,
        VideoStabilizationModel,
        video_stabilization_loss,
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
        max_jitter=2,
        noise_std=0.01,
    )
    ds = SyntheticVideoStabilizationDataset(cfg)
    jittered, target = ds[0]

    assert tuple(jittered.shape) == (6, 1, 32, 32)
    assert set(target.keys()) == {"stabilized"}
    assert tuple(target["stabilized"].shape) == (6, 1, 32, 32)
    assert jittered.dtype == torch.float32
    assert target["stabilized"].dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    batch_jittered, batch_targets = next(iter(train_loader))
    assert tuple(batch_jittered.shape) == (4, 6, 1, 32, 32)
    assert tuple(batch_targets["stabilized"].shape) == (4, 6, 1, 32, 32)

    model = VideoStabilizationModel(
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
        )
    )
    outputs = model(batch_jittered)
    assert set(outputs.keys()) == {"stabilized"}
    assert tuple(outputs["stabilized"].shape) == (4, 6, 1, 32, 32)

    loss, parts = video_stabilization_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"reconstruction_loss"}
    assert float(parts["reconstruction_loss"]) >= 0.0
    loss.backward()


def test_vision_video_stabilization_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_67_synthetic_video_stabilization.data import DataConfig
    from tracks.vision.lesson_67_synthetic_video_stabilization.model import ModelConfig
    from tracks.vision.lesson_67_synthetic_video_stabilization.train import (
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
            run_name="pytest_video_stabilization_smoke",
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
            max_jitter=2,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_67_synthetic_video_stabilization" / "pytest_video_stabilization_smoke"
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
    for key in ("train_loss", "train_reconstruction_loss", "eval_loss", "eval_reconstruction_loss"):
        assert key in record
        assert float(record[key]) >= 0.0
