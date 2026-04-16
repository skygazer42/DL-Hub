import json
import os

import pytest

torch = pytest.importorskip("torch")


def test_masked_language_modeling_batch_contract() -> None:
    from tracks.nlp.lesson_13_toy_masked_language_modeling.data import DataConfig, get_dataloaders

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            max_length=10,
            mask_prob=0.2,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    inputs, targets = next(iter(train_loader))

    assert set(inputs.keys()) == {"input_ids", "attention_mask", "masked_positions"}
    assert tuple(inputs["input_ids"].shape) == (6, 10)
    assert tuple(inputs["attention_mask"].shape) == (6, 10)
    assert tuple(inputs["masked_positions"].shape) == (6, 10)
    assert tuple(targets["labels"].shape) == (6, 10)
    assert vocab.mask_id != vocab.pad_id

    masked_positions = inputs["masked_positions"].to(torch.bool)
    assert torch.any(masked_positions)
    assert torch.all(targets["labels"][~masked_positions] == -100)
    assert torch.all(inputs["input_ids"][masked_positions] == vocab.mask_id)


def test_masked_language_modeling_model_forward_and_loss() -> None:
    from tracks.nlp.lesson_13_toy_masked_language_modeling.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_13_toy_masked_language_modeling.model import (
        ModelConfig,
        ToyMaskedLanguageModel,
        masked_token_accuracy,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            max_length=9,
            mask_prob=0.3,
            val_fraction=0.25,
            seed=1,
            num_workers=0,
        )
    )
    inputs, targets = next(iter(train_loader))
    model = ToyMaskedLanguageModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=9,
            embed_dim=32,
            num_heads=4,
            num_layers=1,
            ff_dim=64,
            dropout=0.0,
        )
    )
    out = model(inputs)

    assert set(out.keys()) == {"logits"}
    assert tuple(out["logits"].shape) == (4, 9, vocab.size)

    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
        out["logits"].reshape(-1, vocab.size),
        targets["labels"].reshape(-1),
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    acc = masked_token_accuracy(out["logits"], targets["labels"], ignore_index=-100)
    assert 0.0 <= acc <= 1.0


def test_masked_language_modeling_training_smoke(tmp_path) -> None:
    from tracks.nlp.lesson_13_toy_masked_language_modeling.data import DataConfig
    from tracks.nlp.lesson_13_toy_masked_language_modeling.train import TrainConfig, run_training

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=1e-3,
                seed=7,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_masked_lm_smoke",
                embed_dim=32,
                num_heads=4,
                num_layers=1,
                ff_dim=64,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=64,
                batch_size=8,
                max_length=10,
                mask_prob=0.2,
                val_fraction=0.25,
                seed=5,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = tmp_path / "outputs" / "nlp" / "lesson_13_toy_masked_language_modeling" / "pytest_masked_lm_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metric_row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metric_row["epoch"] == 1
    assert metric_row["train_loss"] >= 0.0
    assert 0.0 <= metric_row["train_masked_acc"] <= 1.0
    assert metric_row["eval_loss"] >= 0.0
    assert 0.0 <= metric_row["eval_masked_acc"] <= 1.0
