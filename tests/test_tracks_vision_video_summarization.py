import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_video_summarization_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_71_synthetic_video_summarization.data import (
        DataConfig,
        SyntheticVideoSummarizationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_71_synthetic_video_summarization.model import (
        ModelConfig,
        VideoSummarizationModel,
        frame_importance_mae,
        video_summarization_loss,
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
        noise_std=0.01,
    )
    ds = SyntheticVideoSummarizationDataset(cfg)
    clip, target = ds[0]

    assert tuple(clip.shape) == (6, 1, 32, 32)
    assert set(target.keys()) == {"importance"}
    assert tuple(target["importance"].shape) == (6,)
    assert clip.dtype == torch.float32
    assert target["importance"].dtype == torch.float32
    assert 0.0 <= float(target["importance"].min().item()) <= float(target["importance"].max().item()) <= 1.0

    train_loader, _ = get_dataloaders(cfg)
    batch_clips, batch_targets = next(iter(train_loader))
    assert tuple(batch_clips.shape) == (4, 6, 1, 32, 32)
    assert tuple(batch_targets["importance"].shape) == (4, 6)

    model = VideoSummarizationModel(
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
            seq_len=6,
        )
    )
    outputs = model(batch_clips)
    assert set(outputs.keys()) == {"importance_logits"}
    assert tuple(outputs["importance_logits"].shape) == (4, 6)

    loss, parts = video_summarization_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"importance_loss"}
    assert float(parts["importance_loss"]) >= 0.0
    assert 0.0 <= frame_importance_mae(outputs["importance_logits"], batch_targets["importance"]) <= 1.0
    loss.backward()


def test_vision_video_summarization_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_71_synthetic_video_summarization.data import DataConfig
    from tracks.vision.lesson_71_synthetic_video_summarization.model import ModelConfig
    from tracks.vision.lesson_71_synthetic_video_summarization.train import (
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
            run_name="pytest_video_summarization_smoke",
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
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=1,
            hidden_channels=16,
            num_blocks=3,
            seq_len=6,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_71_synthetic_video_summarization" / "pytest_video_summarization_smoke"
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
    for key in ("train_loss", "train_importance_loss", "train_mae", "eval_loss", "eval_importance_loss", "eval_mae"):
        assert key in record
        assert float(record[key]) >= 0.0
