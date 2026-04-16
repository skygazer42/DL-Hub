import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_face_landmark_detection_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_32_synthetic_face_landmark_detection.data import (
        DataConfig,
        SyntheticFaceLandmarkDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_32_synthetic_face_landmark_detection.model import (
        FaceLandmarkRegressor,
        ModelConfig,
        landmark_regression_loss,
        mean_landmark_l2_pixels,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=48,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=1,
        num_landmarks=5,
    )

    ds = SyntheticFaceLandmarkDataset(cfg)
    image, landmarks = ds[0]
    assert tuple(image.shape) == (1, 48, 48)
    assert tuple(landmarks.shape) == (10,)
    assert image.dtype == torch.float32
    assert landmarks.dtype == torch.float32
    assert torch.all(landmarks >= 0.0)
    assert torch.all(landmarks <= 1.0)

    train_loader, _ = get_dataloaders(cfg)
    images, landmark_targets = next(iter(train_loader))
    assert tuple(images.shape) == (4, 1, 48, 48)
    assert tuple(landmark_targets.shape) == (4, 10)

    model = FaceLandmarkRegressor(
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, num_landmarks=5)
    )
    outputs = model(images)
    assert tuple(outputs.shape) == (4, 10)

    loss = landmark_regression_loss(outputs, landmark_targets)
    assert torch.isfinite(loss)
    assert mean_landmark_l2_pixels(outputs.detach(), landmark_targets, image_size=48) >= 0.0
    loss.backward()


def test_face_landmark_detection_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_32_synthetic_face_landmark_detection.data import DataConfig
    from tracks.vision.lesson_32_synthetic_face_landmark_detection.model import ModelConfig
    from tracks.vision.lesson_32_synthetic_face_landmark_detection.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_face_landmark_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=48,
            val_fraction=0.2,
            seed=5,
            num_workers=0,
            in_channels=1,
            num_landmarks=5,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, num_landmarks=5),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_32_synthetic_face_landmark_detection"
        / "pytest_face_landmark_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "logs" / "train.log").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(metrics) == 1
    record = metrics[0]
    for key in ("train_mse", "eval_mse", "eval_l2_px"):
        assert key in record
        assert float(record[key]) >= 0.0
