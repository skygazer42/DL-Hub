import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_toy_video_to_video_data_and_model_contract() -> None:
    from tracks.generative.lesson_50_toy_video_to_video.data import DataConfig, get_dataloaders
    from tracks.generative.lesson_50_toy_video_to_video.model import (
        ModelConfig,
        ToyVideoToVideoModel,
        video_to_video_loss,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            image_size=16,
            num_frames=4,
            seed=0,
            num_workers=0,
            val_fraction=0.25,
        )
    )
    source_video, target_video = next(iter(train_loader))

    assert tuple(source_video.shape) == (4, 3, 4, 16, 16)
    assert tuple(target_video.shape) == (4, 3, 4, 16, 16)

    model = ToyVideoToVideoModel(
        ModelConfig(
            in_channels=3,
            family="diffusion_v2v",
            variant="diffusion_v2v_tiny",
            width_mult=1.0,
        )
    )
    outputs = model(source_video)

    assert set(outputs.keys()) == {"video", "residual", "mix"}
    assert tuple(outputs["video"].shape) == (4, 3, 4, 16, 16)
    assert tuple(outputs["residual"].shape) == (4, 3, 4, 16, 16)
    assert tuple(outputs["mix"].shape) == (4, 3, 4, 16, 16)

    loss, parts = video_to_video_loss(outputs, target_video)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"video_loss", "residual_reg"}
    assert float(parts["video_loss"]) >= 0.0
    assert float(parts["residual_reg"]) >= 0.0
    loss.backward()


def test_toy_video_to_video_training_and_dry_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.generative.lesson_50_toy_video_to_video.data import DataConfig
    from tracks.generative.lesson_50_toy_video_to_video.model import ModelConfig
    from tracks.generative.lesson_50_toy_video_to_video.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=50,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_video_to_video_smoke",
        ),
        DataConfig(
            num_samples=36,
            batch_size=4,
            image_size=16,
            num_frames=4,
            seed=11,
            num_workers=0,
            val_fraction=0.25,
        ),
        ModelConfig(
            in_channels=3,
            family="diffusion_v2v",
            variant="diffusion_v2v_tiny",
            width_mult=1.0,
        ),
    )

    assert exit_code == 0
    run_dir = tmp_path / "generative" / "lesson_50_toy_video_to_video" / "pytest_video_to_video_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(metrics) == 1
    record = metrics[0]
    for key in ("train_loss", "train_video_loss", "eval_loss"):
        assert key in record
        assert float(record[key]) >= 0.0

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_50_toy_video_to_video",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.generative.lesson_50_toy_video_to_video.train" in proc.stdout
