import os

import pytest

torch = pytest.importorskip("torch")


def test_llm_tool_calling_batch_and_loss_smoke() -> None:
    from tracks.llm.lesson_13_toy_tool_calling_agent.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_13_toy_tool_calling_agent.model import ModelConfig, ToyToolCallingAgent
    from tracks.llm.lesson_13_toy_tool_calling_agent.train import tool_calling_loss

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
        "tool_targets",
    }
    assert tuple(batch["input_ids"].shape) == (8, 20)
    assert tuple(batch["labels"].shape) == (8, 20)
    assert tuple(batch["tool_targets"].shape) == (8,)

    model = ToyToolCallingAgent(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=20,
            num_tools=vocab.num_tools,
            embed_dim=48,
            hidden_dim=64,
            dropout=0.0,
        )
    )
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    assert set(outputs.keys()) == {"token_logits", "tool_logits"}
    assert tuple(outputs["token_logits"].shape) == (8, 20, vocab.size)
    assert tuple(outputs["tool_logits"].shape) == (8, vocab.num_tools)

    loss = tool_calling_loss(
        token_logits=outputs["token_logits"],
        tool_logits=outputs["tool_logits"],
        labels=batch["labels"],
        tool_targets=batch["tool_targets"],
        ignore_index=vocab.ignore_index,
        tool_loss_weight=0.5,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_llm_tool_calling_training_smoke(tmp_path) -> None:
    from tracks.llm.lesson_13_toy_tool_calling_agent.data import DataConfig
    from tracks.llm.lesson_13_toy_tool_calling_agent.train import TrainConfig, run_training

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=2e-3,
                tool_loss_weight=0.5,
                seed=19,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_tool_calling_smoke",
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

    run_dir = (
        tmp_path / "outputs" / "llm" / "lesson_13_toy_tool_calling_agent" / "pytest_tool_calling_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
