"""Multimodal track smoke checks (torch-only)."""


def check_clip_compact_retrieval() -> None:
    from dlhub.paths import build_run_paths
    from tracks.multimodal.lesson_01_clip_compact_retrieval.data import DataConfig
    from tracks.multimodal.lesson_01_clip_compact_retrieval.train import TrainConfig, run_training

    run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
        ),
        DataConfig(
            num_samples=32,
            batch_size=8,
            image_size=16,
            max_text_length=6,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        ),
    )

    paths = build_run_paths(
        track="multimodal", lesson="lesson_01_clip_compact_retrieval", run_name="smoke"
    )
    assert (paths.run_dir / "config.json").is_file()
    assert (paths.run_dir / "metrics.jsonl").is_file()
    assert (paths.checkpoints_dir / "checkpoint.pt").is_file()
