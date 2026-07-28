import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_llm_reward_modeling_pairwise_batch_and_loss() -> None:
    from tracks.llm.lesson_07_compact_reward_modeling.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_07_compact_reward_modeling.model import ModelConfig, CompactRewardModel

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=18,
            base_vocab_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {
        "chosen_input_ids",
        "chosen_attention_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
    }
    assert tuple(batch["chosen_input_ids"].shape) == (8, 18)
    assert tuple(batch["chosen_attention_mask"].shape) == (8, 18)
    assert tuple(batch["rejected_input_ids"].shape) == (8, 18)
    assert tuple(batch["rejected_attention_mask"].shape) == (8, 18)

    model = CompactRewardModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=48,
            hidden_dim=64,
            dropout=0.0,
        )
    )
    chosen_rewards = model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
    rejected_rewards = model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
    assert tuple(chosen_rewards.shape) == (8,)
    assert tuple(rejected_rewards.shape) == (8,)

    loss = model.preference_loss(chosen_rewards=chosen_rewards, rejected_rewards=rejected_rewards)
    assert torch.isfinite(loss)
    loss.backward()


def test_llm_reward_modeling_training_smoke() -> None:
    from tracks.llm.lesson_07_compact_reward_modeling.data import DataConfig
    from tracks.llm.lesson_07_compact_reward_modeling.train import TrainConfig, run_training

    run_dir = (
        _repo_root()
        / "outputs"
        / "llm"
        / "lesson_07_compact_reward_modeling"
        / "pytest_reward_modeling_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=11,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_reward_modeling_smoke",
            embed_dim=48,
            hidden_dim=64,
            dropout=0.0,
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=18,
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
