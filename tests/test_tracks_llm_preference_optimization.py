import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_llm_preference_optimization_shapes_and_loss_smoke() -> None:
    from tracks.llm.lesson_06_compact_preference_optimization.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_06_compact_preference_optimization.model import (
        ModelConfig,
        PreferenceTransformerLM,
    )
    from tracks.llm.lesson_06_compact_preference_optimization.train import preference_dpo_loss

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=20,
            base_vocab_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))
    assert tuple(batch["chosen_input_ids"].shape) == (8, 20)
    assert tuple(batch["rejected_input_ids"].shape) == (8, 20)
    assert tuple(batch["chosen_labels"].shape) == (8, 20)
    assert tuple(batch["rejected_labels"].shape) == (8, 20)
    assert (batch["chosen_labels"] != vocab.ignore_index).any()
    assert (batch["rejected_labels"] != vocab.ignore_index).any()

    policy_model = PreferenceTransformerLM(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=20,
            embed_dim=48,
            num_heads=4,
            num_layers=2,
            ff_dim=96,
            dropout=0.0,
        )
    )
    reference_model = PreferenceTransformerLM(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=20,
            embed_dim=48,
            num_heads=4,
            num_layers=2,
            ff_dim=96,
            dropout=0.0,
        )
    )
    for p in reference_model.parameters():
        p.requires_grad_(False)

    chosen_inputs = {
        "input_ids": batch["chosen_input_ids"],
        "attention_mask": batch["chosen_attention_mask"],
    }
    rejected_inputs = {
        "input_ids": batch["rejected_input_ids"],
        "attention_mask": batch["rejected_attention_mask"],
    }
    chosen_policy = policy_model(chosen_inputs)
    rejected_policy = policy_model(rejected_inputs)
    with torch.no_grad():
        chosen_ref = reference_model(chosen_inputs)
        rejected_ref = reference_model(rejected_inputs)

    loss = preference_dpo_loss(
        chosen_policy_logits=chosen_policy,
        rejected_policy_logits=rejected_policy,
        chosen_ref_logits=chosen_ref,
        rejected_ref_logits=rejected_ref,
        chosen_labels=batch["chosen_labels"],
        rejected_labels=batch["rejected_labels"],
        beta=0.5,
        ignore_index=vocab.ignore_index,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_llm_preference_optimization_training_smoke() -> None:
    from tracks.llm.lesson_06_compact_preference_optimization.data import DataConfig
    from tracks.llm.lesson_06_compact_preference_optimization.train import TrainConfig, run_training

    run_dir = (
        _repo_root()
        / "outputs"
        / "llm"
        / "lesson_06_compact_preference_optimization"
        / "pytest_preference_optimization_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=7,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_preference_optimization_smoke",
            embed_dim=48,
            num_heads=4,
            num_layers=2,
            ff_dim=96,
            dropout=0.0,
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=20,
            base_vocab_size=32,
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
