import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_diffusion_image_editing_dataloaders_smoke() -> None:
    from tracks.generative.lesson_10_toy_diffusion_image_editing.data import DataConfig, get_dataloaders

    train_loader, val_loader = get_dataloaders(
        DataConfig(num_samples=48, batch_size=8, image_size=28, seed=0, num_workers=0, val_fraction=0.25)
    )
    source, target, mask, control_token = next(iter(train_loader))
    val_source, val_target, val_mask, val_control = next(iter(val_loader))

    assert source.shape == (8, 1, 28, 28)
    assert target.shape == (8, 1, 28, 28)
    assert mask.shape == (8, 1, 28, 28)
    assert control_token.shape == (8,)
    assert val_source.shape == (8, 1, 28, 28)
    assert val_target.shape == (8, 1, 28, 28)
    assert val_mask.shape == (8, 1, 28, 28)
    assert val_control.shape == (8,)
    assert torch.all(mask >= 0.0)
    assert torch.all(mask <= 1.0)
    assert torch.all((mask == 0.0) | (mask == 1.0))


def test_diffusion_image_editing_model_pipeline_smoke() -> None:
    from tracks.generative.lesson_10_toy_diffusion_image_editing.model import (
        DiffusionSchedule,
        ModelConfig,
        ToyDiffusionImageEditor,
        q_sample,
    )

    cfg = ModelConfig(image_size=28, in_channels=1, hidden_channels=24, control_vocab_size=2)
    schedule = DiffusionSchedule(num_steps=12)
    model = ToyDiffusionImageEditor(cfg)

    source = torch.rand((4, 1, 28, 28), dtype=torch.float32)
    target = torch.rand((4, 1, 28, 28), dtype=torch.float32)
    mask = (torch.rand((4, 1, 28, 28), dtype=torch.float32) > 0.6).to(torch.float32)
    control_token = torch.randint(low=0, high=2, size=(4,), dtype=torch.long)

    noise = torch.randn_like(target)
    timesteps = torch.randint(low=0, high=schedule.num_steps, size=(4,), dtype=torch.long)
    xt = q_sample(schedule, target, timesteps, noise)
    pred_noise = model(xt, source, mask, timesteps, control_token)
    edited = model.sample_edit(
        schedule,
        source=source,
        mask=mask,
        control_token=control_token,
        device=torch.device("cpu"),
        num_steps=6,
    )

    assert xt.shape == target.shape
    assert pred_noise.shape == target.shape
    assert edited.shape == target.shape
    assert torch.all(edited >= 0.0)
    assert torch.all(edited <= 1.0)
    outside_mask = (1.0 - mask).bool()
    assert torch.allclose(edited[outside_mask], source[outside_mask], atol=1e-3)


def test_diffusion_image_editing_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "generative"
        / "lesson_10_toy_diffusion_image_editing"
        / "pytest_diffusion_image_editing_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.generative.lesson_10_toy_diffusion_image_editing.train",
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
            "pytest_diffusion_image_editing_smoke",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "edited_samples.pt").is_file()
    assert (run_dir / "denoise_grid.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
