import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_compact_diffusion_inpainting_data_and_model_contract() -> None:
    from tracks.generative.lesson_14_compact_diffusion_inpainting.data import DataConfig, get_dataloaders
    from tracks.generative.lesson_14_compact_diffusion_inpainting.model import (
        DiffusionSchedule,
        ModelConfig,
        CompactInpaintingDiffusionModel,
        q_sample,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=48, batch_size=6, image_size=28, seed=0, num_workers=0, val_fraction=0.25)
    )
    context_image, target_image, mask = next(iter(train_loader))
    assert tuple(context_image.shape) == (6, 1, 28, 28)
    assert tuple(target_image.shape) == (6, 1, 28, 28)
    assert tuple(mask.shape) == (6, 1, 28, 28)
    assert torch.all((mask == 0.0) | (mask == 1.0))

    cfg = ModelConfig(image_size=28, in_channels=1, hidden_channels=16)
    schedule = DiffusionSchedule(num_steps=12)
    model = CompactInpaintingDiffusionModel(cfg)

    noise = torch.randn_like(target_image)
    timesteps = torch.randint(low=0, high=schedule.num_steps, size=(6,), dtype=torch.long)
    xt = q_sample(schedule, target_image, timesteps, noise)
    pred_noise = model(xt=xt, context=context_image, mask=mask, timesteps=timesteps)
    sampled = model.sample(
        schedule=schedule,
        context=context_image,
        mask=mask,
        device=torch.device("cpu"),
        num_steps=6,
    )

    assert tuple(pred_noise.shape) == (6, 1, 28, 28)
    assert tuple(sampled.shape) == (6, 1, 28, 28)
    assert torch.all(sampled >= 0.0)
    assert torch.all(sampled <= 1.0)
    known_region = (1.0 - mask).bool()
    assert torch.allclose(sampled[known_region], context_image[known_region], atol=1e-3)


def test_compact_diffusion_inpainting_training_and_dry_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.generative.lesson_14_compact_diffusion_inpainting.data import DataConfig
    from tracks.generative.lesson_14_compact_diffusion_inpainting.model import DiffusionSchedule, ModelConfig
    from tracks.generative.lesson_14_compact_diffusion_inpainting.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=7,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_inpainting_smoke",
            num_sample_steps=6,
        ),
        DataConfig(num_samples=64, batch_size=8, image_size=28, seed=3, num_workers=0, val_fraction=0.25),
        ModelConfig(image_size=28, in_channels=1, hidden_channels=16),
        DiffusionSchedule(num_steps=12),
    )

    assert exit_code == 0
    run_dir = tmp_path / "generative" / "lesson_14_compact_diffusion_inpainting" / "pytest_inpainting_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "denoise_grid.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_14_compact_diffusion_inpainting",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.generative.lesson_14_compact_diffusion_inpainting.train" in proc.stdout
