import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_compact_text_to_3d_data_and_model_contract() -> None:
    from tracks.generative.lesson_47_compact_text_to_3d.data import DataConfig, get_dataloaders
    from tracks.generative.lesson_47_compact_text_to_3d.model import (
        ModelConfig,
        CompactTextTo3DModel,
        text_to_3d_loss,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=40,
            batch_size=5,
            seed=0,
            num_workers=0,
            val_fraction=0.25,
        )
    )
    text_features, targets = next(iter(train_loader))

    assert tuple(text_features.shape) == (5, 32)
    assert set(targets.keys()) == {"density", "mesh_tokens"}
    assert tuple(targets["density"].shape) == (5, 1, 12, 12, 12)
    assert tuple(targets["mesh_tokens"].shape) == (5, 12, 3)

    model = CompactTextTo3DModel(
        ModelConfig(
            in_channels=3,
            latent_dim=64,
            family="dreamfusion_baseline",
            variant="dreamfusion_baseline_tiny",
            width_mult=0.5,
        )
    )
    outputs = model(text_features)

    assert set(outputs.keys()) == {"triplanes", "density", "mesh_tokens"}
    assert tuple(outputs["triplanes"].shape) == (5, 3, 3, 12, 12)
    assert tuple(outputs["density"].shape) == (5, 1, 12, 12, 12)
    assert tuple(outputs["mesh_tokens"].shape) == (5, 12, 3)

    loss, parts = text_to_3d_loss(outputs, targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"density_loss", "mesh_loss"}
    assert float(parts["density_loss"]) >= 0.0
    assert float(parts["mesh_loss"]) >= 0.0
    loss.backward()


def test_compact_text_to_3d_training_and_dry_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.generative.lesson_47_compact_text_to_3d.data import DataConfig
    from tracks.generative.lesson_47_compact_text_to_3d.model import ModelConfig
    from tracks.generative.lesson_47_compact_text_to_3d.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=47,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_text_to_3d_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=6,
            seed=7,
            num_workers=0,
            val_fraction=0.25,
        ),
        ModelConfig(
            in_channels=3,
            latent_dim=64,
            family="dreamfusion_baseline",
            variant="dreamfusion_baseline_tiny",
            width_mult=0.5,
        ),
    )

    assert exit_code == 0
    run_dir = tmp_path / "generative" / "lesson_47_compact_text_to_3d" / "pytest_text_to_3d_smoke"
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
    for key in ("train_loss", "train_density_loss", "train_mesh_loss", "eval_loss"):
        assert key in record
        assert float(record[key]) >= 0.0

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_47_compact_text_to_3d",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.generative.lesson_47_compact_text_to_3d.train" in proc.stdout
