import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_weak_supervision_batch_contract() -> None:
    from tracks.nlp.lesson_22_compact_weak_supervision_text_classification.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.nlp.lesson_22_compact_weak_supervision_text_classification.model import (
        ModelConfig,
        WeakSupervisionTextClassifier,
        weak_supervision_accuracy,
        weak_supervision_loss,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            max_length=12,
            num_labeling_functions=3,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {
        "input_ids",
        "attention_mask",
        "lf_votes",
        "lf_mask",
        "label_probs",
        "gold_labels",
    }
    assert tuple(batch["input_ids"].shape) == (6, 12)
    assert tuple(batch["lf_votes"].shape) == (6, 3)
    assert tuple(batch["lf_mask"].shape) == (6, 3)
    assert tuple(batch["label_probs"].shape) == (6, 2)
    assert tuple(batch["gold_labels"].shape) == (6,)

    model = WeakSupervisionTextClassifier(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=24,
            hidden_dim=16,
            num_labeling_functions=3,
            dropout=0.0,
        )
    )
    logits = model(batch)
    assert tuple(logits.shape) == (6, 2)

    loss = weak_supervision_loss(logits, batch["label_probs"])
    assert torch.isfinite(loss)
    assert 0.0 <= weak_supervision_accuracy(logits, batch["gold_labels"]) <= 1.0


def test_weak_supervision_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_22_compact_weak_supervision_text_classification.data import DataConfig
    from tracks.nlp.lesson_22_compact_weak_supervision_text_classification.train import (
        TrainConfig,
        run_training,
    )

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=2e-3,
                seed=7,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_weak_supervision_smoke",
                embed_dim=24,
                hidden_dim=16,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=64,
                batch_size=8,
                max_length=12,
                num_labeling_functions=3,
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
        / "nlp"
        / "lesson_22_compact_weak_supervision_text_classification"
        / "pytest_weak_supervision_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_acc"] <= 1.0
