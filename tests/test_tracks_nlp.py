import pytest

torch = pytest.importorskip("torch")


def test_nlp_lesson_01_shapes_smoke() -> None:
    from tracks.nlp.lesson_01_compact_text_classification.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_01_compact_text_classification.model import (
        MeanPoolTextClassifier,
        ModelConfig,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64, batch_size=8, max_length=16, val_fraction=0.2, seed=0, num_workers=0
        )
    )
    inputs, y = next(iter(train_loader))

    assert set(inputs.keys()) == {"input_ids", "attention_mask"}
    assert tuple(inputs["input_ids"].shape) == (8, 16)
    assert tuple(inputs["attention_mask"].shape) == (8, 16)
    assert tuple(y.shape) == (8,)

    model = MeanPoolTextClassifier(
        ModelConfig(
            vocab_size=vocab.size, pad_id=vocab.pad_id, embed_dim=32, num_classes=2, dropout=0.1
        )
    )
    logits = model(inputs)
    assert tuple(logits.shape) == (8, 2)


def test_nlp_lesson_02_transformer_shapes_smoke() -> None:
    from tracks.nlp.lesson_02_compact_text_classification_transformer.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.nlp.lesson_02_compact_text_classification_transformer.model import (
        ModelConfig,
        TransformerTextClassifier,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64, batch_size=8, max_length=16, val_fraction=0.2, seed=0, num_workers=0
        )
    )
    inputs, y = next(iter(train_loader))

    assert set(inputs.keys()) == {"input_ids", "attention_mask"}
    assert tuple(inputs["input_ids"].shape) == (8, 16)
    assert tuple(inputs["attention_mask"].shape) == (8, 16)
    assert tuple(y.shape) == (8,)

    model = TransformerTextClassifier(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            max_length=16,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            ff_dim=64,
            dropout=0.1,
            num_classes=2,
        )
    )
    logits = model(inputs)
    assert tuple(logits.shape) == (8, 2)


def test_nlp_lesson_03_ner_shapes_smoke() -> None:
    from tracks.nlp.lesson_03_compact_ner_bilstm.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_03_compact_ner_bilstm.model import BiLstmNerTagger, ModelConfig

    train_loader, _, vocab, tag_vocab = get_dataloaders(
        DataConfig(
            num_samples=64, batch_size=8, max_length=16, val_fraction=0.2, seed=0, num_workers=0
        )
    )
    inputs, y = next(iter(train_loader))
    assert set(inputs.keys()) == {"input_ids", "attention_mask"}
    assert tuple(inputs["input_ids"].shape) == (8, 16)
    assert tuple(inputs["attention_mask"].shape) == (8, 16)
    assert tuple(y.shape) == (8, 16)

    model = BiLstmNerTagger(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=32,
            hidden_dim=64,
            num_tags=tag_vocab.size,
            dropout=0.1,
        )
    )
    logits = model(inputs)
    assert tuple(logits.shape) == (8, 16, tag_vocab.size)


def test_nlp_lesson_07_reading_comprehension_shapes_smoke() -> None:
    from tracks.nlp.lesson_07_reading_comprehension.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_07_reading_comprehension.model import ModelConfig, SimpleSpanQA

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            context_length=16,
            question_length=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        )
    )
    inputs, targets = next(iter(train_loader))

    assert set(inputs.keys()) == {"context_ids", "context_mask", "question_ids", "question_mask"}
    assert tuple(inputs["context_ids"].shape) == (8, 16)
    assert tuple(inputs["context_mask"].shape) == (8, 16)
    assert tuple(inputs["question_ids"].shape) == (8, 4)
    assert tuple(inputs["question_mask"].shape) == (8, 4)

    assert set(targets.keys()) == {"start", "end"}
    assert tuple(targets["start"].shape) == (8,)
    assert tuple(targets["end"].shape) == (8,)

    model = SimpleSpanQA(
        ModelConfig(
            vocab_size=vocab.size, pad_id=vocab.pad_id, embed_dim=32, hidden_dim=32, dropout=0.1
        )
    )
    out = model(**inputs)
    assert set(out.keys()) == {"start_logits", "end_logits"}
    assert tuple(out["start_logits"].shape) == (8, 16)
    assert tuple(out["end_logits"].shape) == (8, 16)


def test_nlp_lesson_04_seq2seq_attention_shapes_smoke() -> None:
    from tracks.nlp.lesson_04_compact_seq2seq_attention_generation.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.nlp.lesson_04_compact_seq2seq_attention_generation.model import (
        ModelConfig,
        Seq2SeqWithAttention,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=8,
            min_len=4,
            max_len=8,
            base_vocab_size=16,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        )
    )
    inputs, targets = next(iter(train_loader))
    assert set(inputs.keys()) == {"src_ids", "src_mask", "tgt_in_ids", "tgt_mask"}
    assert set(targets.keys()) == {"tgt_out_ids"}

    model = Seq2SeqWithAttention(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            embed_dim=32,
            hidden_dim=64,
            dropout=0.1,
        )
    )
    out = model(
        src_ids=inputs["src_ids"], src_mask=inputs["src_mask"], tgt_in_ids=inputs["tgt_in_ids"]
    )
    assert set(out.keys()) == {"logits", "attn"}
    assert tuple(out["logits"].shape) == (8, 9, vocab.size)

    loss = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)(
        out["logits"].reshape(-1, vocab.size), targets["tgt_out_ids"].reshape(-1)
    )
    assert torch.isfinite(loss)


def test_nlp_lesson_05_textcnn_shapes_smoke() -> None:
    from tracks.nlp.lesson_05_compact_text_classification_textcnn.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.nlp.lesson_05_compact_text_classification_textcnn.model import (
        ModelConfig,
        TextCNNClassifier,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64, batch_size=8, max_length=16, val_fraction=0.2, seed=0, num_workers=0
        )
    )
    inputs, y = next(iter(train_loader))

    model = TextCNNClassifier(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=32,
            num_classes=2,
            dropout=0.1,
            num_filters=16,
        )
    )
    logits = model(inputs)
    assert tuple(logits.shape) == (8, 2)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    assert torch.isfinite(loss)


def test_nlp_lesson_06_bilstm_shapes_smoke() -> None:
    from tracks.nlp.lesson_06_compact_text_classification_bilstm.data import DataConfig, get_dataloaders
    from tracks.nlp.lesson_06_compact_text_classification_bilstm.model import (
        BiLSTMTextClassifier,
        ModelConfig,
    )

    train_loader, _, vocab = get_dataloaders(
        DataConfig(
            num_samples=64, batch_size=8, max_length=16, val_fraction=0.2, seed=0, num_workers=0
        )
    )
    inputs, y = next(iter(train_loader))

    model = BiLSTMTextClassifier(
        ModelConfig(
            vocab_size=vocab.size,
            pad_id=vocab.pad_id,
            embed_dim=32,
            hidden_dim=16,
            num_classes=2,
            dropout=0.1,
        )
    )
    logits = model(inputs)
    assert tuple(logits.shape) == (8, 2)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    assert torch.isfinite(loss)
