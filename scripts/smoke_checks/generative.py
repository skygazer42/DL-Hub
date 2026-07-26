"""Generative track smoke checks (torch-only)."""


def run() -> None:
    from dlhub.paths import build_run_paths

    # 4.13) Generative lesson: VAE (fake, torch-only).
    from tracks.generative.lesson_01_vae_mnist.data import DataConfig as VaeData
    from tracks.generative.lesson_01_vae_mnist.model import ModelConfig as VaeModel
    from tracks.generative.lesson_01_vae_mnist.train import TrainConfig as VaeTrain
    from tracks.generative.lesson_01_vae_mnist.train import run_training as run_vae

    run_vae(
        VaeTrain(
            epochs=1,
            learning_rate=1e-3,
            beta=1.0,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
        ),
        VaeData(
            dataset="fake",
            batch_size=64,
            num_workers=0,
            num_samples=256,
            seed=0,
            val_fraction=0.2,
        ),
        VaeModel(latent_dim=8, hidden_dim=64),
    )

    vae_paths = build_run_paths(
        track="generative", lesson="lesson_01_vae_mnist", run_name="smoke"
    )
    assert (vae_paths.run_dir / "config.json").is_file()
    assert (vae_paths.run_dir / "metrics.jsonl").is_file()
    assert (vae_paths.run_dir / "samples.pt").is_file()
    assert (vae_paths.run_dir / "recons.pt").is_file()
    assert (vae_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.14) Generative lesson: GAN (fake, torch-only).
    from tracks.generative.lesson_02_gan_mnist.data import DataConfig as GanData
    from tracks.generative.lesson_02_gan_mnist.model import ModelConfig as GanModel
    from tracks.generative.lesson_02_gan_mnist.train import TrainConfig as GanTrain
    from tracks.generative.lesson_02_gan_mnist.train import run_training as run_gan

    run_gan(
        GanTrain(
            epochs=1,
            learning_rate=2e-4,
            beta1=0.5,
            beta2=0.999,
            seed=0,
            device="cpu",
            max_train_batches=1,
            run_name="smoke",
            label_smoothing=0.0,
        ),
        GanData(dataset="fake", batch_size=64, num_workers=0, num_samples=256, seed=0),
        GanModel(z_dim=16, hidden_dim=64),
    )

    gan_paths = build_run_paths(
        track="generative", lesson="lesson_02_gan_mnist", run_name="smoke"
    )
    assert (gan_paths.run_dir / "config.json").is_file()
    assert (gan_paths.run_dir / "metrics.jsonl").is_file()
    assert (gan_paths.run_dir / "samples.pt").is_file()
    assert (gan_paths.checkpoints_dir / "checkpoint.pt").is_file()
