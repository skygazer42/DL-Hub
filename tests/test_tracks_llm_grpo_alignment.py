import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_llm_grpo_alignment_batch_and_loss_smoke() -> None:
    from tracks.llm.lesson_10_toy_grpo_alignment.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_10_toy_grpo_alignment.model import ModelConfig, ToyGrpoPolicyLM
    from tracks.llm.lesson_10_toy_grpo_alignment.train import grpo_group_loss

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_prompts=24,
            group_size=4,
            batch_size=3,
            seq_length=20,
            base_vocab_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))
    assert tuple(batch["input_ids"].shape) == (3, 4, 20)
    assert tuple(batch["labels"].shape) == (3, 4, 20)
    assert tuple(batch["response_mask"].shape) == (3, 4, 20)
    assert tuple(batch["group_rewards"].shape) == (3, 4)
    assert (batch["response_mask"].sum(dim=-1) > 0).all()

    policy = ToyGrpoPolicyLM(
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
    inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    logits = policy(inputs)
    loss = grpo_group_loss(
        logits=logits,
        labels=batch["labels"],
        response_mask=batch["response_mask"],
        group_rewards=batch["group_rewards"],
        ignore_index=vocab.ignore_index,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_llm_grpo_alignment_training_smoke() -> None:
    from tracks.llm.lesson_10_toy_grpo_alignment.data import DataConfig
    from tracks.llm.lesson_10_toy_grpo_alignment.train import TrainConfig, run_training

    run_dir = (
        _repo_root()
        / "outputs"
        / "llm"
        / "lesson_10_toy_grpo_alignment"
        / "pytest_grpo_alignment_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=17,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_grpo_alignment_smoke",
            embed_dim=48,
            num_heads=4,
            num_layers=2,
            ff_dim=96,
            dropout=0.0,
        ),
        DataConfig(
            num_prompts=32,
            group_size=4,
            batch_size=4,
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
