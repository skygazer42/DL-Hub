import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_llm_reflection_memory_agent_batch_mask_and_memory_tokens() -> None:
    from tracks.llm.lesson_18_toy_reflection_memory_agent.data import DataConfig, get_dataloaders

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=36,
            base_vocab_size=32,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    inputs, labels = next(iter(train_loader))
    assert tuple(inputs["input_ids"].shape) == (8, 36)
    assert tuple(inputs["attention_mask"].shape) == (8, 36)
    assert tuple(labels.shape) == (8, 36)

    revise_positions = (inputs["input_ids"] == int(vocab.revise_token_id)).to(torch.int64).argmax(dim=1)
    for row_idx, revise_pos in enumerate(revise_positions.tolist()):
        assert torch.all(labels[row_idx, : int(revise_pos)] == int(vocab.ignore_index))
    assert (labels != int(vocab.ignore_index)).any()

    supervised_targets = labels[labels != int(vocab.ignore_index)]
    assert torch.all(supervised_targets != int(vocab.user_token_id))
    assert torch.all(supervised_targets != int(vocab.assistant_token_id))
    assert torch.all(supervised_targets != int(vocab.reflect_token_id))
    assert torch.all(supervised_targets != int(vocab.memory_write_token_id))
    assert torch.all(supervised_targets != int(vocab.memory_read_token_id))
    assert torch.all(supervised_targets != int(vocab.revise_token_id))

    write_count = (inputs["input_ids"] == int(vocab.memory_write_token_id)).sum().item()
    read_count = (inputs["input_ids"] == int(vocab.memory_read_token_id)).sum().item()
    assert write_count >= 8
    assert read_count >= 8


def test_llm_reflection_memory_agent_model_and_generation_smoke() -> None:
    from tracks.llm.lesson_18_toy_reflection_memory_agent.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_18_toy_reflection_memory_agent.model import (
        ModelConfig,
        ReflectionMemoryTransformerLM,
    )
    from tracks.llm.lesson_18_toy_reflection_memory_agent.train import generate_response

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            seq_length=36,
            base_vocab_size=16,
            val_fraction=0.25,
            seed=1,
            num_workers=0,
        )
    )
    inputs, _ = next(iter(train_loader))
    model = ReflectionMemoryTransformerLM(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=36,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            ff_dim=64,
            dropout=0.0,
        )
    )

    logits = model(inputs)
    assert tuple(logits.shape) == (4, 36, vocab.size)

    topic_id = int(vocab.content_start_id)
    prompt = [
        int(vocab.system_token_id),
        int(vocab.user_token_id),
        topic_id,
        int(vocab.assistant_token_id),
        topic_id + 2,
        int(vocab.reflect_token_id),
        topic_id + 1,
        int(vocab.memory_write_token_id),
        topic_id + 1,
        int(vocab.memory_read_token_id),
        topic_id,
        int(vocab.revise_token_id),
        int(vocab.assistant_token_id),
    ]
    generated = generate_response(
        model=model,
        device=torch.device("cpu"),
        prompt_ids=prompt,
        max_new_tokens=6,
        pad_id=vocab.pad_id,
        stop_id=vocab.eos_id,
    )
    assert generated[: len(prompt)] == prompt
    assert len(generated) >= len(prompt)
    assert len(generated) <= len(prompt) + 6


def test_llm_reflection_memory_agent_training_smoke(tmp_path) -> None:
    from tracks.llm.lesson_18_toy_reflection_memory_agent.data import DataConfig
    from tracks.llm.lesson_18_toy_reflection_memory_agent.train import TrainConfig, run_training

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=2e-3,
                seed=21,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_reflection_memory_agent_smoke",
                generation_tokens=10,
                embed_dim=32,
                num_heads=4,
                num_layers=2,
                ff_dim=64,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=80,
                batch_size=8,
                seq_length=36,
                base_vocab_size=32,
                val_fraction=0.25,
                seed=4,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = (
        tmp_path
        / "outputs"
        / "llm"
        / "lesson_18_toy_reflection_memory_agent"
        / "pytest_reflection_memory_agent_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert metric_row["eval_loss"] >= 0.0
