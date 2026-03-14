import shutil
import subprocess
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_llava_batch_shapes() -> None:
    from tracks.multimodal.lesson_03_llava_toy_instruction_vlm.data import (
        DataConfig,
        get_dataloaders,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=16,
        max_text_length=12,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 16, 16)
    assert batch["instruction_ids"].shape == (8, 12)
    assert batch["input_ids"].shape == (8, 12)
    assert batch["labels"].shape == (8, 12)
    assert batch["attention_mask"].shape == (8, 12)
    assert len(batch["question_type"]) == 8
    assert "what" in vocab.token_to_id
    assert "where" in vocab.token_to_id
    assert "yes" in vocab.token_to_id
    assert "no" in vocab.token_to_id
    assert "top" in vocab.token_to_id
    assert "left" in vocab.token_to_id


def test_multimodal_llava_model_outputs() -> None:
    from tracks.multimodal.lesson_03_llava_toy_instruction_vlm.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.multimodal.lesson_03_llava_toy_instruction_vlm.model import (
        ModelConfig,
        ToyLLaVAModel,
        qa_loss,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=16,
        max_text_length=12,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyLLaVAModel(
        ModelConfig(
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
    assert outputs["logits"].shape == (8, 12, vocab.size)
    assert outputs["visual_tokens"].shape[0] == 8
    assert outputs["visual_tokens"].shape[-1] == 48

    loss = qa_loss(outputs["logits"], batch["labels"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_multimodal_llava_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_03_llava_toy_instruction_vlm"
        / "pytest_llava_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_03_llava_toy_instruction_vlm.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "16",
            "--max-text-length",
            "12",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_llava_smoke",
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
