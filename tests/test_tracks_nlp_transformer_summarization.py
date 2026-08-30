import pytest

torch = pytest.importorskip("torch")


def test_transformer_summarization_batch_contract_and_teacher_forcing() -> None:
    from tracks.nlp.lesson_09_compact_transformer_summarization.data import DataConfig, get_dataloaders

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=8,
            min_len=6,
            max_len=10,
            base_vocab_size=16,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    inputs, targets = next(iter(train_loader))

    assert set(inputs.keys()) == {"src_ids", "src_mask", "tgt_in_ids", "tgt_mask"}
    assert set(targets.keys()) == {"tgt_out_ids"}

    assert tuple(inputs["src_ids"].shape) == (8, 10)
    assert tuple(inputs["src_mask"].shape) == (8, 10)
    assert tuple(inputs["tgt_in_ids"].shape)[0] == 8
    assert tuple(targets["tgt_out_ids"].shape) == tuple(inputs["tgt_in_ids"].shape)

    tgt_mask = inputs["tgt_mask"].to(torch.bool)
    assert torch.all(inputs["tgt_in_ids"][:, 0] == vocab.bos_id)
    assert torch.all(inputs["tgt_in_ids"][:, 1:][tgt_mask[:, 1:]] == targets["tgt_out_ids"][:, :-1][tgt_mask[:, 1:]])

    eos_counts = (targets["tgt_out_ids"] == vocab.eos_id).sum(dim=1)
    assert torch.all(eos_counts == 1)
    assert torch.all(inputs["src_mask"].sum(dim=1) >= inputs["tgt_mask"].sum(dim=1))


def test_transformer_summarization_model_is_causal_and_decodes() -> None:
    from tracks.nlp.lesson_09_compact_transformer_summarization.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_09_compact_transformer_summarization.model import (
        ModelConfig,
        CompactTransformerSummarizer,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            min_len=6,
            max_len=9,
            base_vocab_size=12,
            val_fraction=0.25,
            seed=1,
            num_workers=0,
        )
    )
    inputs, targets = next(iter(train_loader))
    model = CompactTransformerSummarizer(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_src_len=9,
            max_tgt_len=inputs["tgt_in_ids"].shape[1],
            embed_dim=32,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            ff_dim=64,
            dropout=0.0,
        )
    )
    assert model.encoder.enable_nested_tensor is False
    assert model.encoder.use_nested_tensor is False
    model.eval()

    out = model(
        src_ids=inputs["src_ids"],
        src_mask=inputs["src_mask"],
        tgt_in_ids=inputs["tgt_in_ids"],
    )
    assert set(out.keys()) == {"logits"}
    assert tuple(out["logits"].shape) == (
        4,
        inputs["tgt_in_ids"].shape[1],
        vocab.size,
    )

    loss = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)(
        out["logits"].reshape(-1, vocab.size),
        targets["tgt_out_ids"].reshape(-1),
    )
    assert torch.isfinite(loss)

    alt_tgt_in_ids = inputs["tgt_in_ids"].clone()
    alt_tgt_in_ids[:, -1] = vocab.eos_id
    alt_out = model(
        src_ids=inputs["src_ids"],
        src_mask=inputs["src_mask"],
        tgt_in_ids=alt_tgt_in_ids,
    )
    torch.testing.assert_close(out["logits"][:, :-1], alt_out["logits"][:, :-1])

    pred_ids = model.greedy_decode(
        src_ids=inputs["src_ids"],
        src_mask=inputs["src_mask"],
        max_len=inputs["tgt_in_ids"].shape[1],
    )
    assert tuple(pred_ids.shape) == tuple(inputs["tgt_in_ids"].shape)
    assert pred_ids.dtype == torch.long


def test_transformer_summarization_epoch_metrics_are_token_level() -> None:
    from tracks.nlp.lesson_09_compact_transformer_summarization.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_09_compact_transformer_summarization.model import (
        ModelConfig,
        CompactTransformerSummarizer,
    )
    from tracks.nlp.lesson_09_compact_transformer_summarization.train import _run_epoch

    train_loader, val_loader, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            min_len=6,
            max_len=10,
            base_vocab_size=16,
            val_fraction=0.25,
            seed=3,
            num_workers=0,
        )
    )
    model = CompactTransformerSummarizer(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_src_len=10,
            max_tgt_len=6,
            embed_dim=32,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            ff_dim=64,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_stats = _run_epoch(
        model=model,
        loader=train_loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        pad_id=vocab.pad_id,
        max_batches=2,
    )
    eval_stats = _run_epoch(
        model=model,
        loader=val_loader,
        device=torch.device("cpu"),
        optimizer=None,
        pad_id=vocab.pad_id,
        max_batches=1,
    )

    for stats in (train_stats, eval_stats):
        assert torch.isfinite(torch.tensor(stats.loss))
        assert 0.0 <= stats.token_acc <= 1.0
        assert 0.0 <= stats.exact_match <= 1.0
