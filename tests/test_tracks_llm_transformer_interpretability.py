import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_llm_transformer_interpretability_batch_attention_and_saliency_smoke() -> None:
    from tracks.llm.lesson_12_toy_transformer_interpretability.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_12_toy_transformer_interpretability.model import (
        ModelConfig,
        ToyInterpretabilityTransformerLM,
    )
    from tracks.llm.lesson_12_toy_transformer_interpretability.train import (
        compute_attention_map,
        compute_token_saliency,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=16,
            base_vocab_size=24,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))
    assert tuple(batch["input_ids"].shape) == (8, 16)
    assert tuple(batch["labels"].shape) == (8, 16)
    assert tuple(batch["attention_mask"].shape) == (8, 16)
    assert (batch["labels"] != vocab.ignore_index).any()

    model = ToyInterpretabilityTransformerLM(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=16,
            embed_dim=48,
            num_heads=4,
            ff_dim=96,
            dropout=0.0,
        )
    )
    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    logits = model(model_inputs)
    assert tuple(logits.shape) == (8, 16, vocab.size)

    attention = compute_attention_map(model, model_inputs)
    assert tuple(attention.shape) == (8, 4, 16, 16)
    assert torch.isfinite(attention).all()
    row_sums = attention.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    saliency = compute_token_saliency(
        model=model,
        inputs=model_inputs,
        labels=batch["labels"],
        ignore_index=vocab.ignore_index,
    )
    assert tuple(saliency.shape) == (8, 16)
    assert torch.isfinite(saliency).all()
    assert (saliency >= 0).all()
    assert (saliency.sum(dim=-1) > 0).all()


def test_llm_transformer_interpretability_training_smoke() -> None:
    from tracks.llm.lesson_12_toy_transformer_interpretability.data import DataConfig
    from tracks.llm.lesson_12_toy_transformer_interpretability.train import TrainConfig, run_training

    run_dir = (
        _repo_root()
        / "outputs"
        / "llm"
        / "lesson_12_toy_transformer_interpretability"
        / "pytest_transformer_interpretability_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=9,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_transformer_interpretability_smoke",
            embed_dim=48,
            num_heads=4,
            ff_dim=96,
            dropout=0.0,
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=16,
            base_vocab_size=24,
            val_fraction=0.25,
            seed=3,
            num_workers=0,
        ),
    )

    assert exit_code == 0
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
