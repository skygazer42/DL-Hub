import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_llm_instruction_tuning_response_mask_and_shapes() -> None:
    from tracks.llm.lesson_04_compact_instruction_tuning.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_04_compact_instruction_tuning.model import (
        InstructionTransformerLM,
        ModelConfig,
    )

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
    inputs, labels = next(iter(train_loader))
    assert tuple(inputs["input_ids"].shape) == (8, 20)
    assert tuple(inputs["attention_mask"].shape) == (8, 20)
    assert tuple(labels.shape) == (8, 20)
    assert (labels != vocab.ignore_index).any()
    assert torch.all(labels[:, :7] == vocab.ignore_index)

    model = InstructionTransformerLM(
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
    logits = model(inputs)
    assert tuple(logits.shape) == (8, 20, vocab.size)
    loss = torch.nn.CrossEntropyLoss(ignore_index=vocab.ignore_index)(
        logits.reshape(-1, vocab.size), labels.reshape(-1)
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_llm_instruction_tuning_training_smoke() -> None:
    from tracks.llm.lesson_04_compact_instruction_tuning.data import DataConfig
    from tracks.llm.lesson_04_compact_instruction_tuning.train import TrainConfig, run_training

    run_dir = _repo_root() / "outputs" / "llm" / "lesson_04_compact_instruction_tuning" / "pytest_instruction_tuning_smoke"
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
            run_name="pytest_instruction_tuning_smoke",
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
