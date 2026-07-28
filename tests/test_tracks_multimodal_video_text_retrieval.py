import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_video_text_retrieval_batch_shapes() -> None:
    from tracks.multimodal.lesson_17_video_text_retrieval.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        num_frames=6,
        image_size=20,
        max_text_length=12,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["video"].shape == (8, 6, 3, 20, 20)
    assert batch["input_ids"].shape == (8, 12)
    assert batch["attention_mask"].shape == (8, 12)
    assert batch["pair_id"].shape == (8,)
    assert len(batch["caption_text"]) == 8
    assert len(batch["query_text"]) == 8
    assert len(batch["motion_type"]) == 8
    assert "video" in vocab.token_to_id
    assert "moving" in vocab.token_to_id
    assert "left" in vocab.token_to_id
    assert "right" in vocab.token_to_id
    assert "around" in vocab.token_to_id


def test_multimodal_video_text_retrieval_model_outputs() -> None:
    from tracks.multimodal.lesson_17_video_text_retrieval.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_17_video_text_retrieval.model import (
        ModelConfig,
        CompactVideoTextRetrievalModel,
        clip_contrastive_loss,
        recall_at_k,
        retrieval_accuracy,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        num_frames=6,
        image_size=20,
        max_text_length=12,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = CompactVideoTextRetrievalModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_text_length=data_cfg.max_text_length,
            num_frames=data_cfg.num_frames,
            image_size=data_cfg.image_size,
            embed_dim=32,
            vision_width=32,
            text_width=32,
            temporal_width=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {
        "video_embed",
        "text_embed",
        "frame_features",
        "pooled_video_features",
        "logits_per_video",
        "logits_per_text",
    }
    assert outputs["video_embed"].shape == (8, 32)
    assert outputs["text_embed"].shape == (8, 32)
    assert outputs["frame_features"].shape == (8, 6, 32)
    assert outputs["pooled_video_features"].shape == (8, 32)
    assert outputs["logits_per_video"].shape == (8, 8)
    assert outputs["logits_per_text"].shape == (8, 8)

    loss = clip_contrastive_loss(outputs["logits_per_video"], outputs["logits_per_text"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    v2t_acc, t2v_acc = retrieval_accuracy(
        outputs["logits_per_video"], outputs["logits_per_text"]
    )
    v2t_r_at_3, t2v_r_at_3 = recall_at_k(
        outputs["logits_per_video"], outputs["logits_per_text"], k=3
    )
    for metric in (v2t_acc, t2v_acc, v2t_r_at_3, t2v_r_at_3):
        assert 0.0 <= metric <= 1.0


def test_multimodal_video_text_retrieval_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_17_video_text_retrieval"
        / "pytest_video_text_retrieval_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_17_video_text_retrieval.train",
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
            "--max-text-length",
            "12",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_video_text_retrieval_smoke",
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
