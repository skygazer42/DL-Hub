from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_style_transfer_gatys_lesson_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.vision.lesson_15_neural_style_transfer_gatys.train import (
        DataConfig,
        RunConfig,
        run_style_transfer,
    )

    # Keep outputs out of repo outputs/ during tests.
    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    rc = RunConfig(steps=2, lr=0.05, seed=0, device="cpu", run_name="smoke")
    dc = DataConfig(batch_size=2, image_size=32, seed=0)
    run_style_transfer(rc, dc)

    run_dir = tmp_path / "vision" / "lesson_15_neural_style_transfer_gatys" / "smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "stylized.pt").is_file()


def test_vision_style_transfer_cyclegan_lesson_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.vision.lesson_16_style_transfer_translation_cyclegan.train import (
        DataConfig,
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    tc = TrainConfig(epochs=1, learning_rate=1e-3, seed=0, device="cpu", max_train_batches=1, run_name="smoke")
    dc = DataConfig(num_samples=16, batch_size=4, image_size=32, seed=0, num_workers=0)
    run_training(tc, dc)

    run_dir = tmp_path / "vision" / "lesson_16_style_transfer_translation_cyclegan" / "smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

