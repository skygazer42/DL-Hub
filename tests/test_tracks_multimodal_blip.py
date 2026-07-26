import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_blip_batch_shapes() -> None:
    from tracks.multimodal.lesson_02_blip_toy_captioning.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=16,
        max_text_length=10,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        negative_fraction=0.5,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 16, 16)
    assert batch["caption_in_ids"].shape == (8, 10)
    assert batch["caption_out_ids"].shape == (8, 10)
    assert batch["caption_mask"].shape == (8, 10)
    assert batch["itm_input_ids"].shape == (8, 10)
    assert batch["itm_attention_mask"].shape == (8, 10)
    assert batch["itm_label"].shape == (8,)
    assert "a" in vocab.token_to_id
    assert "at" in vocab.token_to_id
    assert "square" in vocab.token_to_id
    assert "top" in vocab.token_to_id


def test_multimodal_blip_model_outputs() -> None:
    from tracks.multimodal.lesson_02_blip_toy_captioning.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_02_blip_toy_captioning.model import (
        ModelConfig,
        ToyBLIPModel,
        blip_lite_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=16,
        max_text_length=10,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
        negative_fraction=0.5,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyBLIPModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_text_length=data_cfg.max_text_length,
            hidden_dim=48,
            vision_width=32,
            embed_dim=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"caption_logits", "itm_logits", "fused_states"}
    assert outputs["caption_logits"].shape == (8, 10, vocab.size)
    assert outputs["itm_logits"].shape == (8, 2)
    assert outputs["fused_states"].shape == (8, 10, 48)

    losses = blip_lite_loss(
        caption_logits=outputs["caption_logits"],
        itm_logits=outputs["itm_logits"],
        caption_targets=batch["caption_out_ids"],
        caption_mask=batch["caption_mask"],
        itm_targets=batch["itm_label"],
        pad_id=vocab.pad_id,
        itm_weight=0.5,
    )
    assert set(losses) >= {"loss", "caption_loss", "itm_loss"}
    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])


def test_multimodal_blip_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_02_blip_toy_captioning"
        / "pytest_blip_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_02_blip_toy_captioning.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "16",
            "--max-text-length",
            "10",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_blip_smoke",
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
