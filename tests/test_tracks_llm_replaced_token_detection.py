import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_llm_replaced_token_detection_batch_model_and_loss_smoke() -> None:
    from tracks.llm.lesson_14_compact_replaced_token_detection_transformer.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.llm.lesson_14_compact_replaced_token_detection_transformer.model import (
        ModelConfig,
        CompactReplacedTokenDetectionTransformer,
    )
    from tracks.llm.lesson_14_compact_replaced_token_detection_transformer.train import (
        replaced_token_detection_loss,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=18,
            base_vocab_size=24,
            replace_probability=0.25,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))
    assert tuple(batch["input_ids"].shape) == (8, 18)
    assert tuple(batch["attention_mask"].shape) == (8, 18)
    assert tuple(batch["labels"].shape) == (8, 18)
    assert tuple(batch["replaced_labels"].shape) == (8, 18)
    assert (batch["labels"] != vocab.ignore_index).any()
    assert batch["replaced_labels"].sum().item() > 0

    model = CompactReplacedTokenDetectionTransformer(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=18,
            embed_dim=48,
            num_heads=4,
            ff_dim=96,
            dropout=0.0,
        )
    )
    outputs = model(
        {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
    )
    assert set(outputs.keys()) == {"token_logits", "rtd_logits"}
    assert tuple(outputs["token_logits"].shape) == (8, 18, vocab.size)
    assert tuple(outputs["rtd_logits"].shape) == (8, 18)

    loss = replaced_token_detection_loss(
        token_logits=outputs["token_logits"],
        rtd_logits=outputs["rtd_logits"],
        labels=batch["labels"],
        replaced_labels=batch["replaced_labels"],
        attention_mask=batch["attention_mask"],
        ignore_index=vocab.ignore_index,
        rtd_loss_weight=0.5,
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_llm_replaced_token_detection_training_smoke() -> None:
    from tracks.llm.lesson_14_compact_replaced_token_detection_transformer.data import DataConfig
    from tracks.llm.lesson_14_compact_replaced_token_detection_transformer.train import (
        TrainConfig,
        run_training,
    )

    run_dir = (
        _repo_root()
        / "outputs"
        / "llm"
        / "lesson_14_compact_replaced_token_detection_transformer"
        / "pytest_replaced_token_detection_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            rtd_loss_weight=0.5,
            seed=11,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_replaced_token_detection_smoke",
            embed_dim=48,
            num_heads=4,
            ff_dim=96,
            dropout=0.0,
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=18,
            base_vocab_size=24,
            replace_probability=0.25,
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
