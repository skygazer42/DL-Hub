"""Foundations track smoke checks (torch-only)."""


def check_linear_regression() -> None:
    # 4.1) Foundations lesson (no downloads, torch-only).
    from dlhub.paths import build_run_paths
    from tracks.foundations.lesson_02_linear_regression_autograd.data import (
        DataConfig as RegData,
    )
    from tracks.foundations.lesson_02_linear_regression_autograd.train import (
        TrainConfig as RegTrain,
    )
    from tracks.foundations.lesson_02_linear_regression_autograd.train import (
        run_training as run_regression,
    )

    run_regression(
        RegTrain(
            epochs=1,
            learning_rate=0.1,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
        ),
        RegData(num_samples=128, batch_size=64, noise_std=0.1),
    )

    reg_paths = build_run_paths(
        track="foundations", lesson="lesson_02_linear_regression_autograd", run_name="smoke"
    )
    assert (reg_paths.run_dir / "config.json").is_file()
    assert (reg_paths.run_dir / "metrics.jsonl").is_file()
    assert (reg_paths.checkpoints_dir / "checkpoint.pt").is_file()
