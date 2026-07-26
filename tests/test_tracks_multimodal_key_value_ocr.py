import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_key_value_ocr_batch_shapes() -> None:
    from tracks.multimodal.lesson_12_key_value_ocr_toy_doc_vlm.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=32,
        max_text_length=20,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 32, 32)
    assert batch["prompt_ids"].shape == (8, 20)
    assert batch["input_ids"].shape == (8, 20)
    assert batch["labels"].shape == (8, 20)
    assert batch["attention_mask"].shape == (8, 20)
    assert batch["present"].shape == (8,)
    assert "read" in vocab.token_to_id
    assert "name" in vocab.token_to_id
    assert "total" in vocab.token_to_id
    assert "none" in vocab.token_to_id
    assert "paid" in vocab.token_to_id


def test_multimodal_key_value_ocr_renders_five_rows() -> None:
    from tracks.multimodal.lesson_12_key_value_ocr_toy_doc_vlm.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=16,
        batch_size=8,
        image_size=32,
        max_text_length=20,
        val_fraction=0.25,
        seed=7,
        num_workers=0,
        min_fields=5,
        max_fields=5,
    )
    train_loader, _val_loader, _vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    bottom_ink = (batch["image"][:, :, 27:, :] < 0.3).any(dim=3).any(dim=2).any(dim=1)
    assert bool(bottom_ink.all())


def test_multimodal_key_value_ocr_model_outputs() -> None:
    from tracks.multimodal.lesson_12_key_value_ocr_toy_doc_vlm.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_12_key_value_ocr_toy_doc_vlm.model import (
        DocOcrModelConfig,
        ToyDocOcrModel,
        ocr_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=32,
        max_text_length=20,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyDocOcrModel(
        DocOcrModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            sep_id=vocab.sep_id,
            max_text_length=data_cfg.max_text_length,
            hidden_dim=48,
            vision_width=32,
            embed_dim=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "visual_tokens"}
    assert outputs["logits"].shape == (8, 20, vocab.size)
    assert outputs["visual_tokens"].shape[0] == 8
    assert outputs["visual_tokens"].shape[-1] == 48

    loss = ocr_loss(outputs["logits"], batch["labels"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_multimodal_key_value_ocr_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_12_key_value_ocr_toy_doc_vlm"
        / "pytest_key_value_ocr_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_12_key_value_ocr_toy_doc_vlm.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "32",
            "--max-text-length",
            "20",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_key_value_ocr_smoke",
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
