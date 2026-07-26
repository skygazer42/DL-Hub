import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_multimodal_prompt_learning_batch_shapes() -> None:
    from tracks.multimodal.lesson_18_prompt_learning_vlm.data import DataConfig, get_dataloaders

    cfg = DataConfig(
        num_samples=48,
        batch_size=8,
        image_size=16,
        max_text_length=8,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    assert batch["image"].shape == (8, 3, 16, 16)
    assert batch["input_ids"].shape == (8, 8)
    assert batch["attention_mask"].shape == (8, 8)
    assert batch["target_index"].shape == (8,)
    assert len(batch["concept_name"]) == 8
    assert "crimson" in vocab.token_to_id
    assert "amber" in vocab.token_to_id
    assert "teal" in vocab.token_to_id
    assert "triangle" in vocab.token_to_id


def test_multimodal_prompt_learning_forward_and_freezing() -> None:
    from tracks.multimodal.lesson_18_prompt_learning_vlm.data import DataConfig, get_dataloaders
    from tracks.multimodal.lesson_18_prompt_learning_vlm.model import (
        PromptLearningConfig,
        ToyPromptLearningVLM,
        clip_contrastive_loss,
        retrieval_accuracy,
    )

    data_cfg = DataConfig(
        num_samples=32,
        batch_size=8,
        image_size=16,
        max_text_length=8,
        val_fraction=0.25,
        seed=1,
        num_workers=0,
    )
    train_loader, _val_loader, vocab = get_dataloaders(data_cfg)
    batch = next(iter(train_loader))

    model = ToyPromptLearningVLM(
        PromptLearningConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_text_length=data_cfg.max_text_length,
            image_size=data_cfg.image_size,
            prompt_length=4,
            embed_dim=32,
            vision_width=32,
            text_width=32,
        )
    )
    outputs = model(batch)

    assert set(outputs) >= {"image_embed", "text_embed", "logits_per_image", "prompt_embed"}
    assert outputs["image_embed"].shape == (8, 32)
    assert outputs["text_embed"].shape == (8, 32)
    assert outputs["logits_per_image"].shape == (8, 8)
    assert outputs["prompt_embed"].shape == (8, 4, 32)

    loss = clip_contrastive_loss(outputs["logits_per_image"], outputs["logits_per_text"])
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    image_to_text, text_to_image = retrieval_accuracy(
        outputs["logits_per_image"], outputs["logits_per_text"]
    )
    assert 0.0 <= image_to_text <= 1.0
    assert 0.0 <= text_to_image <= 1.0

    frozen = [
        *model.vision_encoder.parameters(),
        *model.text_encoder.parameters(),
        *model.image_projection.parameters(),
        *model.text_projection.parameters(),
    ]
    assert frozen
    assert all(not parameter.requires_grad for parameter in frozen)
    assert model.soft_prompt.requires_grad


def test_multimodal_prompt_learning_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "multimodal"
        / "lesson_18_prompt_learning_vlm"
        / "pytest_prompt_learning_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.multimodal.lesson_18_prompt_learning_vlm.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--image-size",
            "16",
            "--max-text-length",
            "8",
            "--prompt-length",
            "4",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_prompt_learning_smoke",
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
