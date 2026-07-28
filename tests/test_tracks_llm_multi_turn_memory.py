import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_llm_lesson_16_multi_turn_memory_data_masks_non_assistant_tokens() -> None:
    from tracks.llm.lesson_16_compact_multi_turn_memory_sft.data import DataConfig, get_dataloaders

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            seq_length=40,
            base_vocab_size=16,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    inputs, labels = next(iter(train_loader))

    assert set(inputs.keys()) == {"input_ids", "attention_mask"}
    assert tuple(inputs["input_ids"].shape) == (4, 40)
    assert tuple(inputs["attention_mask"].shape) == (4, 40)
    assert tuple(labels.shape) == (4, 40)

    supervised = labels != int(vocab.ignore_index)
    assert torch.any(supervised)
    assert torch.all(inputs["attention_mask"][supervised] == 1.0)
    assert torch.all(labels[~supervised] == int(vocab.ignore_index))

    supervised_targets = labels[supervised]
    assert torch.all(supervised_targets != int(vocab.user_token_id))
    assert torch.all(supervised_targets != int(vocab.system_token_id))
    assert torch.all(supervised_targets != int(vocab.assistant_token_id))
    assert torch.all(supervised_targets != int(vocab.memory_token_id))
    assert torch.all(supervised_targets != int(vocab.task_token_id))

    memory_token_count = (inputs["input_ids"] == int(vocab.memory_token_id)).sum().item()
    user_token_count = (inputs["input_ids"] == int(vocab.user_token_id)).sum().item()
    assistant_token_count = (inputs["input_ids"] == int(vocab.assistant_token_id)).sum().item()
    assert memory_token_count >= 8
    assert user_token_count >= 8
    assert assistant_token_count >= 8


def test_llm_lesson_16_multi_turn_memory_model_and_generation_smoke() -> None:
    from tracks.llm.lesson_16_compact_multi_turn_memory_sft.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_16_compact_multi_turn_memory_sft.model import (
        ModelConfig,
        MultiTurnMemoryTransformerLM,
    )
    from tracks.llm.lesson_16_compact_multi_turn_memory_sft.train import generate_reply

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            seq_length=40,
            base_vocab_size=16,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    inputs, _ = next(iter(train_loader))
    model = MultiTurnMemoryTransformerLM(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=40,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            ff_dim=64,
            dropout=0.0,
        )
    )

    logits = model(inputs)
    assert tuple(logits.shape) == (4, 40, vocab.size)

    topic_id = int(vocab.content_start_id)
    prompt = [
        vocab.system_token_id,
        vocab.task_token_id,
        topic_id,
        vocab.memory_token_id,
        topic_id,
        vocab.user_token_id,
        topic_id,
        vocab.assistant_token_id,
    ]
    generated = generate_reply(
        model=model,
        device=torch.device("cpu"),
        prompt_ids=prompt,
        max_new_tokens=5,
        pad_id=vocab.pad_id,
        stop_id=vocab.eos_id,
    )
    assert generated[: len(prompt)] == prompt
    assert len(generated) >= len(prompt)
    assert len(generated) <= len(prompt) + 5


def test_llm_lesson_16_multi_turn_memory_training_smoke(tmp_path) -> None:
    from tracks.llm.lesson_16_compact_multi_turn_memory_sft.data import DataConfig
    from tracks.llm.lesson_16_compact_multi_turn_memory_sft.train import TrainConfig, run_training

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=2e-3,
                seed=123,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="smoke",
                embed_dim=32,
                num_heads=4,
                num_layers=2,
                ff_dim=64,
                dropout=0.0,
                generation_tokens=8,
            ),
            DataConfig(
                num_samples=48,
                batch_size=8,
                seq_length=40,
                base_vocab_size=16,
                val_fraction=0.25,
                seed=7,
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
        / "lesson_16_compact_multi_turn_memory_sft"
        / "smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.jsonl").is_file()
    assert (run_dir / "logs" / "train.log").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert metric_row["eval_loss"] >= 0.0

    sample_row = json.loads((run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert sample_row["epoch"] == 1
    assert "prompt_ids" in sample_row
    assert "gen_ids" in sample_row
    assert len(sample_row["gen_ids"]) >= len(sample_row["prompt_ids"])
