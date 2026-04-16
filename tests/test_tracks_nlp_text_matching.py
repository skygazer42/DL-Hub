import pytest

torch = pytest.importorskip("torch")


def test_nlp_text_matching_shapes_smoke() -> None:
    from tracks.nlp.lesson_08_toy_text_matching_biencoder.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_08_toy_text_matching_biencoder.model import (
        BiEncoderTextMatcher,
        ModelConfig,
        retrieval_accuracy,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            max_query_length=8,
            max_doc_length=12,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        )
    )
    inputs, labels = next(iter(train_loader))

    assert set(inputs.keys()) == {
        "query_input_ids",
        "query_attention_mask",
        "doc_input_ids",
        "doc_attention_mask",
    }
    assert tuple(inputs["query_input_ids"].shape) == (8, 8)
    assert tuple(inputs["query_attention_mask"].shape) == (8, 8)
    assert tuple(inputs["doc_input_ids"].shape) == (8, 12)
    assert tuple(inputs["doc_attention_mask"].shape) == (8, 12)
    assert tuple(labels.shape) == (8,)

    model = BiEncoderTextMatcher(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=32,
            proj_dim=32,
            dropout=0.1,
        )
    )
    outputs = model(inputs)
    assert set(outputs.keys()) == {"query_embeddings", "doc_embeddings", "pair_logits", "sim_matrix"}
    assert tuple(outputs["query_embeddings"].shape) == (8, 32)
    assert tuple(outputs["doc_embeddings"].shape) == (8, 32)
    assert tuple(outputs["pair_logits"].shape) == (8,)
    assert tuple(outputs["sim_matrix"].shape) == (8, 8)

    loss = torch.nn.BCEWithLogitsLoss()(outputs["pair_logits"], labels.to(torch.float32))
    assert torch.isfinite(loss)

    match_acc, retrieval_acc = retrieval_accuracy(
        pair_logits=outputs["pair_logits"],
        sim_matrix=outputs["sim_matrix"],
        labels=labels,
    )
    assert 0.0 <= match_acc <= 1.0
    assert 0.0 <= retrieval_acc <= 1.0
