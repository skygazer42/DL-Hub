import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_reflection_removal_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_24_synthetic_reflection_removal.data import (
        DataConfig,
        SyntheticReflectionRemovalDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_24_synthetic_reflection_removal.model import (
        ModelConfig,
        ReflectionRemovalModel,
        reflection_removal_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        reflection_strength_min=0.15,
        reflection_strength_max=0.55,
        blur_kernel_size=3,
    )
    ds = SyntheticReflectionRemovalDataset(cfg)
    mixture, targets = ds[0]

    assert tuple(mixture.shape) == (3, 32, 32)
    assert set(targets.keys()) == {"transmission", "reflection"}
    assert tuple(targets["transmission"].shape) == (3, 32, 32)
    assert tuple(targets["reflection"].shape) == (3, 32, 32)
    assert mixture.dtype == torch.float32
    assert targets["transmission"].dtype == torch.float32
    assert targets["reflection"].dtype == torch.float32
    assert 0.0 <= float(mixture.min().item()) <= float(mixture.max().item()) <= 1.0

    train_loader, _ = get_dataloaders(cfg)
    batch_mixture, batch_targets = next(iter(train_loader))
    assert tuple(batch_mixture.shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["transmission"].shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["reflection"].shape) == (4, 3, 32, 32)

    model = ReflectionRemovalModel(ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3))
    outputs = model(batch_mixture)
    assert set(outputs.keys()) == {"transmission", "reflection"}
    assert tuple(outputs["transmission"].shape) == (4, 3, 32, 32)
    assert tuple(outputs["reflection"].shape) == (4, 3, 32, 32)

    loss, parts = reflection_removal_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"transmission_loss", "reflection_loss"}
    assert float(parts["transmission_loss"]) >= 0.0
    assert float(parts["reflection_loss"]) >= 0.0
    loss.backward()


def test_vision_reflection_removal_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_24_synthetic_reflection_removal.data import DataConfig
    from tracks.vision.lesson_24_synthetic_reflection_removal.model import ModelConfig
    from tracks.vision.lesson_24_synthetic_reflection_removal.train import (
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
            run_name="pytest_reflection_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=11,
            num_workers=0,
            in_channels=3,
            reflection_strength_min=0.15,
            reflection_strength_max=0.55,
            blur_kernel_size=3,
        ),
        ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path / "vision" / "lesson_24_synthetic_reflection_removal" / "pytest_reflection_smoke"
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
        "train_transmission_loss",
        "train_reflection_loss",
        "eval_loss",
        "eval_transmission_loss",
        "eval_reflection_loss",
        "eval_psnr",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
