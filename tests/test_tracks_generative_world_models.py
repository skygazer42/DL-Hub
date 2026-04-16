import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_toy_world_models_data_and_model_contract() -> None:
    from tracks.generative.lesson_51_toy_world_models.data import DataConfig, get_dataloaders
    from tracks.generative.lesson_51_toy_world_models.model import (
        ModelConfig,
        ToyWorldModelsModel,
        world_models_loss,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=40,
            batch_size=4,
            image_size=16,
            in_channels=3,
            action_dim=4,
            context_dim=12,
            seed=0,
            num_workers=0,
            val_fraction=0.25,
        )
    )
    obs, action, prompt, targets = next(iter(train_loader))

    assert tuple(obs.shape) == (4, 3, 16, 16)
    assert tuple(action.shape) == (4, 4)
    assert tuple(prompt.shape) == (4, 12)
    assert set(targets.keys()) == {"next_obs", "reward", "done"}
    assert tuple(targets["next_obs"].shape) == (4, 3, 16, 16)
    assert tuple(targets["reward"].shape) == (4, 1)
    assert tuple(targets["done"].shape) == (4, 1)

    model = ToyWorldModelsModel(
        ModelConfig(
            in_channels=3,
            action_dim=4,
            context_dim=12,
            family="rssm_world",
            variant="rssm_world_tiny",
            width_mult=1.0,
        )
    )
    outputs = model(obs=obs, action=action, prompt=prompt)

    assert set(outputs.keys()) == {"latent", "next_state", "reward", "done", "reconstruction"}
    assert tuple(outputs["latent"].shape[:1]) == (4,)
    assert tuple(outputs["next_state"].shape[:1]) == (4,)
    assert tuple(outputs["reward"].shape) == (4, 1)
    assert tuple(outputs["done"].shape) == (4, 1)
    assert tuple(outputs["reconstruction"].shape) == (4, 3, 4, 4)

    loss, parts = world_models_loss(outputs, targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"reconstruction_loss", "reward_loss", "done_loss"}
    assert float(parts["reconstruction_loss"]) >= 0.0
    assert float(parts["reward_loss"]) >= 0.0
    assert float(parts["done_loss"]) >= 0.0
    loss.backward()


def test_toy_world_models_training_and_dry_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.generative.lesson_51_toy_world_models.data import DataConfig
    from tracks.generative.lesson_51_toy_world_models.model import ModelConfig
    from tracks.generative.lesson_51_toy_world_models.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=51,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_world_models_smoke",
            family="rssm_world",
            variant="rssm_world_tiny",
            width_mult=1.0,
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=16,
            in_channels=3,
            action_dim=4,
            context_dim=12,
            seed=5,
            num_workers=0,
            val_fraction=0.25,
        ),
        ModelConfig(
            in_channels=3,
            action_dim=4,
            context_dim=12,
            family="rssm_world",
            variant="rssm_world_tiny",
            width_mult=1.0,
        ),
    )

    assert exit_code == 0
    run_dir = tmp_path / "generative" / "lesson_51_toy_world_models" / "pytest_world_models_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(metrics) == 1
    record = metrics[0]
    for key in ("train_loss", "train_reward_mae", "eval_loss", "eval_reward_mae"):
        assert key in record
        assert float(record[key]) >= 0.0

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_51_toy_world_models",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.generative.lesson_51_toy_world_models.train" in proc.stdout
