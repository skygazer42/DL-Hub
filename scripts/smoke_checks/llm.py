"""LLM track smoke checks (torch-only)."""


def check_compact_causal_lm() -> None:
    # 4.3) LLM lesson: compact causal LM (torch-only).
    from dlhub.paths import build_run_paths
    from tracks.llm.lesson_01_compact_causal_lm_transformer.data import DataConfig as LmData
    from tracks.llm.lesson_01_compact_causal_lm_transformer.train import TrainConfig as LmTrain
    from tracks.llm.lesson_01_compact_causal_lm_transformer.train import run_training as run_lm

    run_lm(
        LmTrain(
            epochs=1,
            learning_rate=2e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
            dropout=0.1,
        ),
        LmData(
            num_samples=256,
            batch_size=32,
            seq_length=32,
            base_vocab_size=64,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    lm_paths = build_run_paths(
        track="llm", lesson="lesson_01_compact_causal_lm_transformer", run_name="smoke"
    )
    assert (lm_paths.run_dir / "config.json").is_file()
    assert (lm_paths.run_dir / "metrics.jsonl").is_file()
    assert (lm_paths.run_dir / "vocab.json").is_file()
    assert (lm_paths.run_dir / "samples.jsonl").is_file()
    assert (lm_paths.checkpoints_dir / "checkpoint.pt").is_file()
