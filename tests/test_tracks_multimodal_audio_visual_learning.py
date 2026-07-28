import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_audio_visual_learning_batch_shapes() -> None:
    from tracks.multimodal.lesson_20_audio_visual_learning.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        num_frames=5,
        image_size=20,
        num_mel_bins=24,
        num_audio_steps=12,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["video"].shape == (8, 5, 3, 20, 20)
    assert batch["audio_spectrogram"].shape == (8, 1, 24, 12)
    assert batch["event_id"].shape == (8,)
    assert batch["motion_id"].shape == (8,)
    assert len(batch["event_name"]) == 8
    assert len(batch["motion_name"]) == 8
    assert len(batch["audio_pattern"]) == 8


def test_multimodal_audio_visual_learning_model_outputs() -> None:
    from tracks.multimodal.lesson_20_audio_visual_learning.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_20_audio_visual_learning.model import (
        AudioVisualLearningConfig,
        CompactAudioVisualLearningModel,
        clip_contrastive_loss,
        retrieval_accuracy,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        num_frames=5,
        image_size=20,
        num_mel_bins=24,
        num_audio_steps=12,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = CompactAudioVisualLearningModel(
        AudioVisualLearningConfig(
            num_frames=data_cfg.num_frames,
            image_size=data_cfg.image_size,
            num_mel_bins=data_cfg.num_mel_bins,
            num_audio_steps=data_cfg.num_audio_steps,
            num_events=6,
            embed_dim=32,
            vision_width=32,
            audio_width=32,
            fusion_width=48,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {
        "video_embed",
        "audio_embed",
        "fused_embed",
        "logits_per_video",
        "logits_per_audio",
        "event_logits",
        "motion_logits",
    }
    assert outputs["video_embed"].shape == (8, 32)
    assert outputs["audio_embed"].shape == (8, 32)
    assert outputs["fused_embed"].shape == (8, 48)
    assert outputs["logits_per_video"].shape == (8, 8)
    assert outputs["logits_per_audio"].shape == (8, 8)
    assert outputs["event_logits"].shape == (8, 6)
    assert outputs["motion_logits"].shape == (8, 4)

    loss = clip_contrastive_loss(outputs["logits_per_video"], outputs["logits_per_audio"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    v2a, a2v = retrieval_accuracy(outputs["logits_per_video"], outputs["logits_per_audio"])
    assert 0.0 <= v2a <= 1.0
    assert 0.0 <= a2v <= 1.0


def test_multimodal_audio_visual_learning_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_20_audio_visual_learning"
        / "pytest_audio_visual_learning_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_20_audio_visual_learning.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--num-frames",
            "5",
            "--image-size",
            "20",
            "--num-mel-bins",
            "24",
            "--num-audio-steps",
            "12",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_audio_visual_learning_smoke",
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
