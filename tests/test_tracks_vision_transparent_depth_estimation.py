import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_transparent_depth_estimation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_77_synthetic_transparent_depth_estimation.data import (
        DataConfig,
        SyntheticTransparentDepthDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_77_synthetic_transparent_depth_estimation.model import (
        ModelConfig,
        TransparentDepthEstimator,
        transparent_depth_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        glass_strength_min=0.20,
        glass_strength_max=0.50,
        noise_std=0.01,
    )
    ds = SyntheticTransparentDepthDataset(cfg)
    image, targets = ds[0]

    assert tuple(image.shape) == (3, 32, 32)
    assert set(targets.keys()) == {"depth", "transparency"}
    assert tuple(targets["depth"].shape) == (1, 32, 32)
    assert tuple(targets["transparency"].shape) == (1, 32, 32)
    assert image.dtype == torch.float32
    assert targets["depth"].dtype == torch.float32
    assert targets["transparency"].dtype == torch.float32
    assert 0.0 <= float(targets["transparency"].min().item()) <= float(
        targets["transparency"].max().item()
    ) <= 1.0

    train_loader, _ = get_dataloaders(cfg)
    batch_image, batch_targets = next(iter(train_loader))
    assert tuple(batch_image.shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["depth"].shape) == (4, 1, 32, 32)
    assert tuple(batch_targets["transparency"].shape) == (4, 1, 32, 32)

    model = TransparentDepthEstimator(
        ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3)
    )
    outputs = model(batch_image)
    assert set(outputs.keys()) == {"depth", "transparency"}
    assert tuple(outputs["depth"].shape) == (4, 1, 32, 32)
    assert tuple(outputs["transparency"].shape) == (4, 1, 32, 32)

    loss, parts = transparent_depth_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"depth_loss", "mask_loss"}
    assert float(parts["depth_loss"]) >= 0.0
    assert float(parts["mask_loss"]) >= 0.0
    loss.backward()


def test_vision_transparent_depth_estimation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_77_synthetic_transparent_depth_estimation.data import DataConfig
    from tracks.vision.lesson_77_synthetic_transparent_depth_estimation.model import ModelConfig
    from tracks.vision.lesson_77_synthetic_transparent_depth_estimation.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_transparent_depth_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=11,
            num_workers=0,
            in_channels=3,
            glass_strength_min=0.20,
            glass_strength_max=0.50,
            noise_std=0.01,
        ),
        ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_77_synthetic_transparent_depth_estimation"
        / "pytest_transparent_depth_smoke"
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
    for key in (
        "train_loss",
        "train_depth_mae",
        "train_mask_bce",
        "eval_loss",
        "eval_depth_mae",
        "eval_mask_bce",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
