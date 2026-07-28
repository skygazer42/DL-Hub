import os

import pytest

torch = pytest.importorskip("torch")


def test_llm_rag_language_model_batch_and_loss_smoke() -> None:
    from tracks.llm.lesson_11_compact_rag_language_model.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_11_compact_rag_language_model.model import ModelConfig, CompactRagLanguageModel

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
        "doc_ids",
    }
    assert tuple(batch["input_ids"].shape) == (8, 20)
    assert tuple(batch["labels"].shape) == (8, 20)
    assert tuple(batch["doc_ids"].shape) == (8,)

    model = CompactRagLanguageModel(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=20,
            num_docs=vocab.num_docs,
            embed_dim=48,
            hidden_dim=64,
        )
    )
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        doc_ids=batch["doc_ids"],
    )
    assert tuple(logits.shape) == (8, 20, vocab.size)

    loss = torch.nn.CrossEntropyLoss(ignore_index=vocab.ignore_index)(
        logits.reshape(-1, vocab.size),
        batch["labels"].reshape(-1),
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()


def test_llm_rag_language_model_training_smoke(tmp_path) -> None:
    from tracks.llm.lesson_11_compact_rag_language_model.data import DataConfig
    from tracks.llm.lesson_11_compact_rag_language_model.train import TrainConfig, run_training

    os.environ["DLHUB_OUTPUTS_DIR"] = str(tmp_path / "outputs")
    try:
        exit_code = run_training(
            TrainConfig(
                epochs=1,
                learning_rate=2e-3,
                seed=11,
                device="cpu",
                max_train_batches=2,
                max_eval_batches=1,
                run_name="pytest_rag_lm_smoke",
                embed_dim=48,
                hidden_dim=64,
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
        tmp_path
        / "outputs"
        / "llm"
        / "lesson_11_compact_rag_language_model"
        / "pytest_rag_lm_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "vocab.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
