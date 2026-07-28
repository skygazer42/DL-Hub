"""NLP track smoke checks (torch-only)."""


def run() -> None:
    from dlhub.paths import build_run_paths

    # 4.11) NLP lesson (compact, torch-only).
    from tracks.nlp.lesson_01_compact_text_classification.data import DataConfig as NlpData
    from tracks.nlp.lesson_01_compact_text_classification.train import TrainConfig as NlpTrain
    from tracks.nlp.lesson_01_compact_text_classification.train import run_training as run_nlp

    run_nlp(
        NlpTrain(
            epochs=1,
            learning_rate=1e-2,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            embed_dim=32,
            dropout=0.1,
        ),
        NlpData(
            num_samples=256,
            batch_size=32,
            max_length=16,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    nlp_paths = build_run_paths(
        track="nlp", lesson="lesson_01_compact_text_classification", run_name="smoke"
    )
    assert (nlp_paths.run_dir / "config.json").is_file()
    assert (nlp_paths.run_dir / "metrics.jsonl").is_file()
    assert (nlp_paths.run_dir / "vocab.json").is_file()
    assert (nlp_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.12) NLP lesson: TextCNN (compact, torch-only).
    from tracks.nlp.lesson_05_compact_text_classification_textcnn.data import DataConfig as CnnData
    from tracks.nlp.lesson_05_compact_text_classification_textcnn.train import (
        TrainConfig as CnnTrain,
    )
    from tracks.nlp.lesson_05_compact_text_classification_textcnn.train import (
        run_training as run_cnn,
    )

    run_cnn(
        CnnTrain(
            epochs=1,
            learning_rate=2e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            embed_dim=32,
            dropout=0.2,
            num_filters=32,
        ),
        CnnData(
            num_samples=256,
            batch_size=32,
            max_length=16,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    cnn_paths = build_run_paths(
        track="nlp", lesson="lesson_05_compact_text_classification_textcnn", run_name="smoke"
    )
    assert (cnn_paths.run_dir / "config.json").is_file()
    assert (cnn_paths.run_dir / "metrics.jsonl").is_file()
    assert (cnn_paths.run_dir / "vocab.json").is_file()
    assert (cnn_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.13) NLP lesson: BiLSTM classifier (compact, torch-only).
    from tracks.nlp.lesson_06_compact_text_classification_bilstm.data import DataConfig as RnnData
    from tracks.nlp.lesson_06_compact_text_classification_bilstm.train import (
        TrainConfig as RnnTrain,
    )
    from tracks.nlp.lesson_06_compact_text_classification_bilstm.train import (
        run_training as run_rnn,
    )

    run_rnn(
        RnnTrain(
            epochs=1,
            learning_rate=2e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            embed_dim=32,
            hidden_dim=32,
            dropout=0.2,
        ),
        RnnData(
            num_samples=256,
            batch_size=32,
            max_length=16,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    rnn_paths = build_run_paths(
        track="nlp", lesson="lesson_06_compact_text_classification_bilstm", run_name="smoke"
    )
    assert (rnn_paths.run_dir / "config.json").is_file()
    assert (rnn_paths.run_dir / "metrics.jsonl").is_file()
    assert (rnn_paths.run_dir / "vocab.json").is_file()
    assert (rnn_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.10) NLP lesson: transformer encoder (compact, torch-only).
    from tracks.nlp.lesson_02_compact_text_classification_transformer.data import (
        DataConfig as NlpTrData,
    )
    from tracks.nlp.lesson_02_compact_text_classification_transformer.train import (
        TrainConfig as NlpTrTrain,
    )
    from tracks.nlp.lesson_02_compact_text_classification_transformer.train import (
        run_training as run_nlp_tr,
    )

    run_nlp_tr(
        NlpTrTrain(
            epochs=1,
            learning_rate=3e-4,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            ff_dim=64,
            dropout=0.1,
        ),
        NlpTrData(
            num_samples=256,
            batch_size=32,
            max_length=16,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    nlp_tr_paths = build_run_paths(
        track="nlp", lesson="lesson_02_compact_text_classification_transformer", run_name="smoke"
    )
    assert (nlp_tr_paths.run_dir / "config.json").is_file()
    assert (nlp_tr_paths.run_dir / "metrics.jsonl").is_file()
    assert (nlp_tr_paths.run_dir / "vocab.json").is_file()
    assert (nlp_tr_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.11) NLP lesson: seq2seq + attention (compact, torch-only).
    from tracks.nlp.lesson_04_compact_seq2seq_attention_generation.data import DataConfig as S2SData
    from tracks.nlp.lesson_04_compact_seq2seq_attention_generation.train import (
        TrainConfig as S2STrain,
    )
    from tracks.nlp.lesson_04_compact_seq2seq_attention_generation.train import (
        run_training as run_s2s,
    )

    run_s2s(
        S2STrain(
            epochs=1,
            learning_rate=2e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            embed_dim=32,
            hidden_dim=64,
            dropout=0.1,
        ),
        S2SData(
            num_samples=256,
            batch_size=32,
            min_len=6,
            max_len=12,
            base_vocab_size=24,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    s2s_paths = build_run_paths(
        track="nlp", lesson="lesson_04_compact_seq2seq_attention_generation", run_name="smoke"
    )
    assert (s2s_paths.run_dir / "config.json").is_file()
    assert (s2s_paths.run_dir / "metrics.jsonl").is_file()
    assert (s2s_paths.run_dir / "vocab.json").is_file()
    assert (s2s_paths.run_dir / "samples.jsonl").is_file()
    assert (s2s_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.11) NLP lesson: compact NER (BiLSTM, torch-only).
    from tracks.nlp.lesson_03_compact_ner_bilstm.data import DataConfig as NerData
    from tracks.nlp.lesson_03_compact_ner_bilstm.train import TrainConfig as NerTrain
    from tracks.nlp.lesson_03_compact_ner_bilstm.train import run_training as run_ner

    run_ner(
        NerTrain(
            epochs=1,
            learning_rate=1e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            embed_dim=32,
            hidden_dim=64,
            dropout=0.1,
        ),
        NerData(
            num_samples=256,
            batch_size=32,
            max_length=16,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    ner_paths = build_run_paths(
        track="nlp", lesson="lesson_03_compact_ner_bilstm", run_name="smoke"
    )
    assert (ner_paths.run_dir / "config.json").is_file()
    assert (ner_paths.run_dir / "metrics.jsonl").is_file()
    assert (ner_paths.run_dir / "vocab.json").is_file()
    assert (ner_paths.run_dir / "tags.json").is_file()
    assert (ner_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.12) NLP lesson: compact reading comprehension (span prediction).
    from tracks.nlp.lesson_07_reading_comprehension.data import DataConfig as RcData
    from tracks.nlp.lesson_07_reading_comprehension.train import TrainConfig as RcTrain
    from tracks.nlp.lesson_07_reading_comprehension.train import run_training as run_rc

    run_rc(
        RcTrain(
            epochs=1,
            learning_rate=3e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            embed_dim=32,
            hidden_dim=32,
            dropout=0.1,
        ),
        RcData(
            num_samples=256,
            batch_size=32,
            context_length=32,
            question_length=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    rc_paths = build_run_paths(
        track="nlp", lesson="lesson_07_reading_comprehension", run_name="smoke"
    )
    assert (rc_paths.run_dir / "config.json").is_file()
    assert (rc_paths.run_dir / "metrics.jsonl").is_file()
    assert (rc_paths.run_dir / "vocab.json").is_file()
    assert (rc_paths.checkpoints_dir / "checkpoint.pt").is_file()
