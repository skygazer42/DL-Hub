import shutil
import subprocess
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_gaze_estimation_batch_shapes() -> None:
    from tracks.multimodal.lesson_30_vision_language_gaze_estimation.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=24,
        heatmap_size=12,
        max_text_length=8,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 24, 24)
    assert batch["head_xy"].shape == (8, 2)
    assert batch["input_ids"].shape == (8, 8)
    assert batch["attention_mask"].shape == (8, 8)
    assert batch["target_point"].shape == (8, 2)
    assert batch["target_heatmap"].shape == (8, 1, 12, 12)
    assert len(batch["prompt_text"]) == 8
    assert "person" in vocab.token_to_id
    assert "looks" in vocab.token_to_id
    assert "left" in vocab.token_to_id
    assert "right" in vocab.token_to_id


def test_multimodal_gaze_estimation_model_outputs() -> None:
    from tracks.multimodal.lesson_30_vision_language_gaze_estimation.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_30_vision_language_gaze_estimation.model import (
        GazeEstimationConfig,
        ToyVisionLanguageGazeEstimator,
        gaze_heatmap_loss,
        gaze_point_l1,
        gaze_point_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=24,
        heatmap_size=12,
        max_text_length=8,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyVisionLanguageGazeEstimator(
        GazeEstimationConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            image_size=data_cfg.image_size,
            max_text_length=data_cfg.max_text_length,
            heatmap_size=data_cfg.heatmap_size,
            hidden_dim=48,
            text_dim=32,
            vision_width=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"gaze_point", "gaze_heatmap"}
    assert outputs["gaze_point"].shape == (8, 2)
    assert outputs["gaze_heatmap"].shape == (8, 1, 12, 12)
    assert torch.all(outputs["gaze_point"] >= 0.0)
    assert torch.all(outputs["gaze_point"] <= 1.0)

    point_loss = gaze_point_loss(outputs["gaze_point"], batch["target_point"])
    heatmap_loss = gaze_heatmap_loss(outputs["gaze_heatmap"], batch["target_heatmap"])
    point_l1 = gaze_point_l1(outputs["gaze_point"], batch["target_point"])
    assert point_loss.ndim == 0
    assert heatmap_loss.ndim == 0
    assert torch.isfinite(point_loss)
    assert torch.isfinite(heatmap_loss)
    assert point_l1 >= 0.0


def test_multimodal_gaze_estimation_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_30_vision_language_gaze_estimation"
        / "pytest_gaze_estimation_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_30_vision_language_gaze_estimation.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "24",
            "--heatmap-size",
            "12",
            "--max-text-length",
            "8",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_gaze_estimation_smoke",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
