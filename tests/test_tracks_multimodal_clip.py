import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_clip_batch_shapes() -> None:
    from tracks.multimodal.lesson_01_clip_compact_retrieval.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=16,
        max_text_length=6,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 16, 16)
    assert batch["input_ids"].shape == (8, 6)
    assert batch["attention_mask"].shape == (8, 6)
    assert batch["pair_id"].shape == (8,)
    assert vocab.pad_id >= 0
    assert "red" in vocab.token_to_id
    assert "square" in vocab.token_to_id
    assert "small" in vocab.token_to_id


def test_multimodal_clip_model_outputs() -> None:
    from tracks.multimodal.lesson_01_clip_compact_retrieval.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_01_clip_compact_retrieval.model import (
        ModelConfig,
        CompactCLIPModel,
        clip_contrastive_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=16,
        max_text_length=6,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = CompactCLIPModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_text_length=data_cfg.max_text_length,
            image_size=data_cfg.image_size,
            embed_dim=32,
            vision_width=32,
            text_width=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {
        "image_embed",
        "text_embed",
        "logits_per_image",
        "logits_per_text",
    }
    assert outputs["image_embed"].shape == (8, 32)
    assert outputs["text_embed"].shape == (8, 32)
    assert outputs["logits_per_image"].shape == (8, 8)
    assert outputs["logits_per_text"].shape == (8, 8)

    loss = clip_contrastive_loss(outputs["logits_per_image"], outputs["logits_per_text"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_multimodal_clip_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_01_clip_compact_retrieval"
        / "pytest_clip_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_01_clip_compact_retrieval.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "16",
            "--max-text-length",
            "6",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_clip_smoke",
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
