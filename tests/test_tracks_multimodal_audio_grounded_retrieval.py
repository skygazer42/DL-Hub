import shutil
import subprocess
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_audio_grounded_retrieval_batch_shapes() -> None:
    from tracks.multimodal.lesson_21_audio_grounded_retrieval.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        num_frames=6,
        image_size=20,
        num_mel_bins=24,
        num_audio_steps=12,
        max_text_length=12,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["video"].shape == (8, 6, 3, 20, 20)
    assert batch["audio_spectrogram"].shape == (8, 1, 24, 12)
    assert batch["input_ids"].shape == (8, 12)
    assert batch["attention_mask"].shape == (8, 12)
    assert batch["pair_id"].shape == (8,)
    assert batch["segment_id"].shape == (8,)
    assert len(batch["query_text"]) == 8
    assert len(batch["event_name"]) == 8
    assert len(batch["motion_name"]) == 8
    assert "audio" in vocab.token_to_id
    assert "video" in vocab.token_to_id
    assert "segment" in vocab.token_to_id
    assert "left" in vocab.token_to_id


def test_multimodal_audio_grounded_retrieval_model_outputs() -> None:
    from tracks.multimodal.lesson_21_audio_grounded_retrieval.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_21_audio_grounded_retrieval.model import (
        AudioGroundedRetrievalConfig,
        ToyAudioGroundedRetrievalModel,
        clip_contrastive_loss,
        retrieval_accuracy,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        num_frames=6,
        image_size=20,
        num_mel_bins=24,
        num_audio_steps=12,
        max_text_length=12,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyAudioGroundedRetrievalModel(
        AudioGroundedRetrievalConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_text_length=data_cfg.max_text_length,
            num_frames=data_cfg.num_frames,
            image_size=data_cfg.image_size,
            num_mel_bins=data_cfg.num_mel_bins,
            num_audio_steps=data_cfg.num_audio_steps,
            embed_dim=32,
            vision_width=32,
            audio_width=32,
            text_width=32,
            fusion_width=48,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {
        "clip_embed",
        "query_embed",
        "video_features",
        "audio_features",
        "fused_features",
        "logits_per_clip",
        "logits_per_query",
    }
    assert outputs["clip_embed"].shape == (8, 32)
    assert outputs["query_embed"].shape == (8, 32)
    assert outputs["video_features"].shape == (8, 32)
    assert outputs["audio_features"].shape == (8, 32)
    assert outputs["fused_features"].shape == (8, 48)
    assert outputs["logits_per_clip"].shape == (8, 8)
    assert outputs["logits_per_query"].shape == (8, 8)

    loss = clip_contrastive_loss(outputs["logits_per_clip"], outputs["logits_per_query"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    c2q, q2c = retrieval_accuracy(outputs["logits_per_clip"], outputs["logits_per_query"])
    assert 0.0 <= c2q <= 1.0
    assert 0.0 <= q2c <= 1.0


def test_multimodal_audio_grounded_retrieval_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_21_audio_grounded_retrieval"
        / "pytest_audio_grounded_retrieval_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_21_audio_grounded_retrieval.train",
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
            "--num-mel-bins",
            "24",
            "--num-audio-steps",
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
            "pytest_audio_grounded_retrieval_smoke",
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
