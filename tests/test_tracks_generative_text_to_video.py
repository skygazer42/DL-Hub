import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_toy_text_to_video_data_and_model_contract() -> None:
    from tracks.generative.lesson_49_toy_text_to_video.data import DataConfig, get_dataloaders
    from tracks.generative.lesson_49_toy_text_to_video.model import (
        ModelConfig,
        ToyTextToVideoModel,
        text_to_video_loss,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=36,
            batch_size=4,
            frames=4,
            seed=0,
            num_workers=0,
            val_fraction=0.25,
        )
    )
    prompts, target_video = next(iter(train_loader))

    assert len(prompts) == 4
    assert tuple(target_video.shape) == (4, 4, 3, 8, 8)

    model = ToyTextToVideoModel(
        ModelConfig(
            in_channels=3,
            family="diffusion_t2v",
            variant="diffusion_t2v_tiny",
            width_mult=1.0,
        )
    )
    outputs = model(prompts)

    assert set(outputs.keys()) == {"video", "prompt_features", "motion"}
    assert tuple(outputs["video"].shape) == (4, 4, 3, 8, 8)
    assert tuple(outputs["prompt_features"].shape) == (4, 32)
    assert tuple(outputs["motion"].shape) == (4, 3, 8, 8)

    loss, parts = text_to_video_loss(outputs, target_video)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"video_loss", "motion_reg"}
    assert float(parts["video_loss"]) >= 0.0
    assert float(parts["motion_reg"]) >= 0.0
    loss.backward()


def test_toy_text_to_video_training_and_dry_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.generative.lesson_49_toy_text_to_video.data import DataConfig
    from tracks.generative.lesson_49_toy_text_to_video.model import ModelConfig
    from tracks.generative.lesson_49_toy_text_to_video.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=49,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_text_to_video_smoke",
        ),
        DataConfig(
            num_samples=40,
            batch_size=5,
            frames=4,
            seed=9,
            num_workers=0,
            val_fraction=0.25,
        ),
        ModelConfig(
            in_channels=3,
            family="diffusion_t2v",
            variant="diffusion_t2v_tiny",
            width_mult=1.0,
        ),
    )

    assert exit_code == 0
    run_dir = tmp_path / "generative" / "lesson_49_toy_text_to_video" / "pytest_text_to_video_smoke"
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
            "lesson_49_toy_text_to_video",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.generative.lesson_49_toy_text_to_video.train" in proc.stdout
