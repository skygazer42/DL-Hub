import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_latent_diffusion_fake_dataloaders_smoke() -> None:
    from tracks.generative.lesson_04_toy_latent_diffusion.data import DataConfig, get_dataloaders

    train_loader, val_loader = get_dataloaders(
        DataConfig(num_samples=48, batch_size=8, image_size=28, seed=0, num_workers=0, val_fraction=0.25)
    )
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert train_batch.shape == (8, 1, 28, 28)
    assert val_batch.shape == (8, 1, 28, 28)
    assert train_batch.dtype == torch.float32
    assert torch.all(train_batch >= 0.0)
    assert torch.all(train_batch <= 1.0)


def test_latent_diffusion_model_pipeline_smoke() -> None:
    from tracks.generative.lesson_04_toy_latent_diffusion.model import (
        LatentDiffusionModel,
        ModelConfig,
        diffusion_loss,
    )

    cfg = ModelConfig(image_size=28, in_channels=1, latent_channels=4, latent_size=7, hidden_channels=16)
    model = LatentDiffusionModel(cfg)
    images = torch.rand((4, 1, 28, 28), dtype=torch.float32)

    latents = model.encode(images)
    assert latents.shape == (4, 4, 7, 7)

    timesteps = torch.randint(low=0, high=cfg.num_diffusion_steps, size=(4,), dtype=torch.long)
    noisy_latents, noise = model.add_noise(latents, timesteps)
    noise_pred = model.predict_noise(noisy_latents, timesteps)
    decoded = model.decode(latents)

    assert noisy_latents.shape == latents.shape
    assert noise.shape == latents.shape
    assert noise_pred.shape == latents.shape
    assert decoded.shape == images.shape

    loss = diffusion_loss(noise_pred=noise_pred, noise=noise, recon_images=decoded, target_images=images)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_latent_diffusion_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "generative"
        / "lesson_04_toy_latent_diffusion"
        / "pytest_latent_diffusion_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.generative.lesson_04_toy_latent_diffusion.train",
            "--epochs",
            "1",
            "--num-samples",
            "48",
            "--batch-size",
            "8",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_latent_diffusion_smoke",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "recons.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
