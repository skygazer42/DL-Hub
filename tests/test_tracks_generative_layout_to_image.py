import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_layout_to_image_batch_and_model_contract() -> None:
    from tracks.generative.lesson_12_compact_layout_to_image.data import DataConfig, get_dataloaders
    from tracks.generative.lesson_12_compact_layout_to_image.model import ModelConfig, CompactLayoutToImageModel

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            image_size=28,
            num_classes=4,
            seed=0,
            num_workers=0,
            val_fraction=0.25,
        )
    )
    layout, image = next(iter(train_loader))
    assert tuple(layout.shape) == (6, 4, 28, 28)
    assert tuple(image.shape) == (6, 1, 28, 28)
    assert torch.all((layout == 0.0) | (layout == 1.0))
    assert torch.all(image >= 0.0)
    assert torch.all(image <= 1.0)

    model = CompactLayoutToImageModel(ModelConfig(num_classes=4, hidden_channels=16))
    logits = model(layout)
    assert tuple(logits.shape) == (6, 1, 28, 28)

    sampled = model.generate(layout)
    assert tuple(sampled.shape) == (6, 1, 28, 28)
    assert torch.all(sampled >= 0.0)
    assert torch.all(sampled <= 1.0)


def test_layout_to_image_training_and_dry_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.generative.lesson_12_compact_layout_to_image.data import DataConfig
    from tracks.generative.lesson_12_compact_layout_to_image.model import ModelConfig
    from tracks.generative.lesson_12_compact_layout_to_image.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=7,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_layout_to_image_smoke",
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            image_size=28,
            num_classes=4,
            seed=3,
            num_workers=0,
            val_fraction=0.25,
        ),
        ModelConfig(num_classes=4, hidden_channels=16),
    )

    assert exit_code == 0
    run_dir = tmp_path / "generative" / "lesson_12_compact_layout_to_image" / "pytest_layout_to_image_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_12_compact_layout_to_image",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.generative.lesson_12_compact_layout_to_image.train" in proc.stdout
