import pytest

torch = pytest.importorskip("torch")


def test_llm_lesson_01_shapes_smoke() -> None:
    from tracks.llm.lesson_01_compact_causal_lm_transformer.data import DataConfig, get_dataloaders
    from tracks.llm.lesson_01_compact_causal_lm_transformer.model import (
        CausalTransformerLM,
        ModelConfig,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            seq_length=16,
            base_vocab_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        )
    )
    inputs, labels = next(iter(train_loader))
    assert set(inputs.keys()) == {"input_ids", "attention_mask"}
    assert tuple(inputs["input_ids"].shape) == (8, 16)
    assert tuple(inputs["attention_mask"].shape) == (8, 16)
    assert tuple(labels.shape) == (8, 16)

    model = CausalTransformerLM(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=16,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
            dropout=0.1,
        )
    )
    logits = model(inputs)
    assert tuple(logits.shape) == (8, 16, vocab.size)

    loss = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)(
        logits.reshape(-1, vocab.size), labels.reshape(-1)
    )
    assert torch.isfinite(loss)
