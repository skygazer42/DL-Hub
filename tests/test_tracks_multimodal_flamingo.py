import shutil
import subprocess
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_flamingo_batch_shapes() -> None:
    from tracks.multimodal.lesson_06_flamingo_toy_interleaved_vlm.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=16,
        max_text_length=28,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["images"].shape == (8, 3, 3, 16, 16)
    assert batch["prompt_ids"].shape == (8, 28)
    assert batch["input_ids"].shape == (8, 28)
    assert batch["labels"].shape == (8, 28)
    assert batch["attention_mask"].shape == (8, 28)
    assert len(batch["task_token"]) == 8
    assert "<image>" in vocab.token_to_id
    assert "example" in vocab.token_to_id
    assert "query" in vocab.token_to_id
    assert "dax" in vocab.token_to_id
    assert "blicket" in vocab.token_to_id


def test_multimodal_flamingo_model_outputs() -> None:
    from tracks.multimodal.lesson_06_flamingo_toy_interleaved_vlm.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_06_flamingo_toy_interleaved_vlm.model import (
        FlamingoModelConfig,
        ToyFlamingoModel,
        qa_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=16,
        max_text_length=28,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyFlamingoModel(
        FlamingoModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            image_token_id=vocab.image_token_id,
            max_text_length=data_cfg.max_text_length,
            hidden_dim=48,
            vision_width=32,
            embed_dim=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"logits", "image_embeddings"}
    assert outputs["logits"].shape == (8, 28, vocab.size)
    assert outputs["image_embeddings"].shape == (8, 3, 48)

    loss = qa_loss(outputs["logits"], batch["labels"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_multimodal_flamingo_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_06_flamingo_toy_interleaved_vlm"
        / "pytest_flamingo_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_06_flamingo_toy_interleaved_vlm.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "16",
            "--max-text-length",
            "28",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_flamingo_smoke",
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
