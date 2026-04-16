import shutil
import subprocess
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_audio_visual_event_localization_batch_shapes() -> None:
    from tracks.multimodal.lesson_22_audio_visual_event_localization.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        num_frames=6,
        image_size=20,
        audio_window=12,
        max_text_length=12,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["video"].shape == (8, 6, 3, 20, 20)
    assert batch["audio_clip"].shape == (8, 6, 12)
    assert batch["query_ids"].shape == (8, 12)
    assert batch["attention_mask"].shape == (8, 12)
    assert batch["event_mask"].shape == (8, 6)
    assert batch["event_frame"].shape == (8,)
    assert len(batch["query_text"]) == 8
    assert len(batch["event_name"]) == 8
    assert len(batch["segment"]) == 8
    assert "when" in vocab.token_to_id
    assert "does" in vocab.token_to_id
    assert "happen" in vocab.token_to_id


def test_multimodal_audio_visual_event_localization_model_outputs() -> None:
    from tracks.multimodal.lesson_22_audio_visual_event_localization.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_22_audio_visual_event_localization.model import (
        AudioVisualEventLocalizationConfig,
        ToyAudioVisualEventLocalizationModel,
        localization_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        num_frames=6,
        image_size=20,
        audio_window=12,
        max_text_length=12,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyAudioVisualEventLocalizationModel(
        AudioVisualEventLocalizationConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            num_frames=data_cfg.num_frames,
            audio_window=data_cfg.audio_window,
            hidden_dim=48,
            vision_width=32,
            audio_width=24,
            text_dim=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"frame_logits", "pred_frame", "frame_probs"}
    assert outputs["frame_logits"].shape == (8, 6)
    assert outputs["frame_probs"].shape == (8, 6)
    assert outputs["pred_frame"].shape == (8,)
    assert outputs["pred_frame"].dtype == torch.long

    losses = localization_loss(outputs["frame_logits"], batch["event_mask"])
    assert losses.ndim == 0
    assert torch.isfinite(losses)


def test_multimodal_audio_visual_event_localization_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_22_audio_visual_event_localization"
        / "pytest_audio_visual_event_localization_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_22_audio_visual_event_localization.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--num-frames",
            "6",
            "--image-size",
            "20",
            "--audio-window",
            "12",
            "--max-text-length",
            "12",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_audio_visual_event_localization_smoke",
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
