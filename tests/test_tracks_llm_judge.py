import os

import pytest

torch = pytest.importorskip("torch")


def test_llm_judge_batch_and_loss_smoke() -> None:
    from tracks.llm.lesson_15_toy_llm_judge.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_15_toy_llm_judge.model import ModelConfig, ToyLlmJudge
    from tracks.llm.lesson_15_toy_llm_judge.train import llm_judge_loss

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=20,
            base_vocab_size=32,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))
    assert set(batch.keys()) == {
        "input_ids",
        "attention_mask",
        "labels",
        "judge_targets",
    }
    assert tuple(batch["input_ids"].shape) == (8, 20)
    assert tuple(batch["labels"].shape) == (8, 20)
    assert tuple(batch["judge_targets"].shape) == (8,)

    model = ToyLlmJudge(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=20,
            embed_dim=48,
            hidden_dim=64,
            dropout=0.0,
        )
    )
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    assert set(outputs.keys()) == {"token_logits", "judge_logits"}
    assert tuple(outputs["token_logits"].shape) == (8, 20, vocab.size)
    assert tuple(outputs["judge_logits"].shape) == (8,)

    loss = llm_judge_loss(
        token_logits=outputs["token_logits"],
        judge_logits=outputs["judge_logits"],
        labels=batch["labels"],
        judge_targets=batch["judge_targets"],
        ignore_index=vocab.ignore_index,
        judge_loss_weight=0.5,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_llm_judge_training_smoke(tmp_path) -> None:
    from tracks.llm.lesson_15_toy_llm_judge.data import DataConfig
    from tracks.llm.lesson_15_toy_llm_judge.train import TrainConfig, run_training

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=2e-3,
                judge_loss_weight=0.5,
                seed=23,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_llm_judge_smoke",
                embed_dim=48,
                hidden_dim=64,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=80,
                batch_size=8,
                seq_length=20,
                base_vocab_size=32,
                val_fraction=0.25,
                seed=3,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = tmp_path / "outputs" / "llm" / "lesson_15_toy_llm_judge" / "pytest_llm_judge_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
