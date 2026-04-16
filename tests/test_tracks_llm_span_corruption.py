import os

import pytest

torch = pytest.importorskip("torch")


def test_llm_span_corruption_batch_contract() -> None:
    from tracks.llm.lesson_08_toy_span_corruption.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_08_toy_span_corruption.model import ModelConfig, ToySpanCorruptionLM

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            seq_length=16,
            base_vocab_size=32,
            mask_ratio=0.25,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    batch = next(iter(train_loader))

    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
    assert tuple(batch["input_ids"].shape) == (6, 16)
    assert tuple(batch["attention_mask"].shape) == (6, 16)
    assert tuple(batch["labels"].shape) == (6, 16)
    assert torch.any(batch["input_ids"] == vocab.mask_id)
    assert torch.any(batch["labels"] != -100)

    model = ToySpanCorruptionLM(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=16,
            embed_dim=32,
            num_heads=4,
            num_layers=1,
            ff_dim=64,
            dropout=0.0,
        )
    )
    outputs = model(batch["input_ids"], batch["attention_mask"])
    assert tuple(outputs.shape) == (6, 16, vocab.size)

    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
        outputs.reshape(-1, vocab.size),
        batch["labels"].reshape(-1),
    )
    assert torch.isfinite(loss)
    loss.backward()


def test_llm_span_corruption_training_smoke(tmp_path) -> None:
    from tracks.llm.lesson_08_toy_span_corruption.data import DataConfig
    from tracks.llm.lesson_08_toy_span_corruption.train import TrainConfig, run_training

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
                run_name="pytest_span_corruption_smoke",
                embed_dim=32,
                num_heads=4,
                num_layers=1,
                ff_dim=64,
                dropout=0.0,
            ),
            DataConfig(
                num_samples=64,
                batch_size=8,
                seq_length=18,
                base_vocab_size=36,
                mask_ratio=0.25,
                val_fraction=0.25,
                seed=5,
                num_workers=0,
            ),
        )
        assert exit_code == 0
    finally:
        os.environ.pop("DLHUB_OUTPUTS_DIR", None)

    run_dir = tmp_path / "outputs" / "llm" / "lesson_08_toy_span_corruption" / "pytest_span_corruption_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()


