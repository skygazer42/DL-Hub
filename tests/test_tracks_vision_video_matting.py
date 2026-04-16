import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_video_matting_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_75_synthetic_video_matting.data import (
        DataConfig,
        SyntheticVideoMattingDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_75_synthetic_video_matting.model import (
        ModelConfig,
        VideoMattingModel,
        video_matting_loss,
        video_matting_mae,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        seq_len=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=1,
    )
    ds = SyntheticVideoMattingDataset(cfg)
    video, trimap, alpha = ds[0]
    assert tuple(video.shape) == (4, 1, 32, 32)
    assert tuple(trimap.shape) == (4, 1, 32, 32)
    assert tuple(alpha.shape) == (4, 1, 32, 32)
    assert video.dtype == torch.float32
    assert trimap.dtype == torch.float32
    assert alpha.dtype == torch.float32
    assert float(alpha.min().item()) >= 0.0
    assert float(alpha.max().item()) <= 1.0

    train_loader, _ = get_dataloaders(cfg)
    videos, trimaps, alphas = next(iter(train_loader))
    assert tuple(videos.shape) == (4, 4, 1, 32, 32)
    assert tuple(trimaps.shape) == (4, 4, 1, 32, 32)
    assert tuple(alphas.shape) == (4, 4, 1, 32, 32)

    model = VideoMattingModel(
        ModelConfig(in_channels=2, hidden_channels=24, num_blocks=3)
    )
    logits = model(videos, trimaps)
    assert tuple(logits.shape) == (4, 4, 1, 32, 32)

    loss, parts = video_matting_loss(logits, alphas)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"bce_loss", "l1_loss"}
    assert float(parts["bce_loss"]) >= 0.0
    assert float(parts["l1_loss"]) >= 0.0
    assert 0.0 <= video_matting_mae(logits.detach(), alphas) <= 1.0
    loss.backward()


def test_vision_video_matting_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_75_synthetic_video_matting.data import DataConfig
    from tracks.vision.lesson_75_synthetic_video_matting.model import ModelConfig
    from tracks.vision.lesson_75_synthetic_video_matting.train import (
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
            run_name="pytest_video_matting_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            seq_len=4,
            image_size=32,
            val_fraction=0.2,
            seed=5,
            num_workers=0,
            in_channels=1,
        ),
        ModelConfig(in_channels=2, hidden_channels=24, num_blocks=3),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_75_synthetic_video_matting" / "pytest_video_matting_smoke"
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
        "train_l1_loss",
        "train_mae",
        "eval_loss",
        "eval_bce_loss",
        "eval_l1_loss",
        "eval_mae",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
