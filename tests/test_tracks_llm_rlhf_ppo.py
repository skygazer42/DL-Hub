import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_llm_rlhf_ppo_batch_and_loss_smoke() -> None:
    from tracks.llm.lesson_09_toy_rlhf_ppo.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_09_toy_rlhf_ppo.model import ModelConfig, ToyPolicyLM, ToyTokenRewardModel
    from tracks.llm.lesson_09_toy_rlhf_ppo.train import ppo_policy_loss

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
    assert set(batch.keys()) == {
        "input_ids",
        "attention_mask",
        "labels",
        "response_mask",
    }
    assert tuple(batch["input_ids"].shape) == (8, 20)
    assert tuple(batch["labels"].shape) == (8, 20)
    assert tuple(batch["response_mask"].shape) == (8, 20)
    assert (batch["response_mask"].sum(dim=1) > 0).all()

    policy = ToyPolicyLM(
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
    reference = ToyPolicyLM(
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
    reward_model = ToyTokenRewardModel(
        pad_id=vocab.pad_id,
        good_token_id=vocab.good_token_id,
        bad_token_id=vocab.bad_token_id,
    )

    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    policy_logits = policy(model_inputs)
    with torch.no_grad():
        ref_logits = reference(model_inputs)
        rewards = reward_model(batch["input_ids"], batch["response_mask"])

    loss = ppo_policy_loss(
        policy_logits=policy_logits,
        reference_logits=ref_logits,
        labels=batch["labels"],
        response_mask=batch["response_mask"],
        rewards=rewards,
        clip_epsilon=0.2,
        kl_coefficient=0.05,
        ignore_index=vocab.ignore_index,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_llm_rlhf_ppo_training_smoke() -> None:
    from tracks.llm.lesson_09_toy_rlhf_ppo.data import DataConfig
    from tracks.llm.lesson_09_toy_rlhf_ppo.train import TrainConfig, run_training

    run_dir = _repo_root() / "outputs" / "llm" / "lesson_09_toy_rlhf_ppo" / "pytest_rlhf_ppo_smoke"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            clip_epsilon=0.2,
            kl_coefficient=0.05,
            seed=13,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_rlhf_ppo_smoke",
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
