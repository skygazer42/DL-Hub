import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_prompt_tuning_batch_contract() -> None:
    from tracks.nlp.lesson_10_compact_prompt_tuning_classifier.data import DataConfig, get_dataloaders

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            max_length=12,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    inputs, labels = next(iter(train_loader))

    assert set(inputs.keys()) == {"input_ids", "attention_mask"}
    assert tuple(inputs["input_ids"].shape) == (8, 12)
    assert tuple(inputs["attention_mask"].shape) == (8, 12)
    assert tuple(labels.shape) == (8,)
    assert vocab.size > 8


def test_prompt_tuning_model_freezes_backbone() -> None:
    from tracks.nlp.lesson_10_compact_prompt_tuning_classifier.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_10_compact_prompt_tuning_classifier.model import (
        ModelConfig,
        PromptTunedTextClassifier,
        trainable_parameter_count,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            max_length=10,
            val_fraction=0.25,
            seed=1,
            num_workers=0,
        )
    )
    inputs, _ = next(iter(train_loader))
    model = PromptTunedTextClassifier(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=10,
            prompt_length=3,
            embed_dim=32,
            num_heads=4,
            num_layers=1,
            ff_dim=64,
            dropout=0.0,
            num_classes=2,
        )
    )
    logits = model(inputs)

    assert tuple(logits.shape) == (4, 2)
    assert model.soft_prompt.requires_grad
    assert not any(parameter.requires_grad for parameter in model.token_embed.parameters())
    assert not any(parameter.requires_grad for parameter in model.blocks.parameters())
    assert trainable_parameter_count(model) > 0


def test_prompt_tuning_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "nlp"
        / "lesson_10_compact_prompt_tuning_classifier"
        / "pytest_prompt_tuning_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.nlp.lesson_10_compact_prompt_tuning_classifier.train",
            "--epochs",
            "1",
            "--num-samples",
            "64",
            "--batch-size",
            "8",
            "--max-length",
            "12",
            "--prompt-length",
            "4",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_prompt_tuning_smoke",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
