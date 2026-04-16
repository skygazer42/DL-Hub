import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_shadow_detection_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_81_synthetic_shadow_detection.data import (
        DataConfig,
        SyntheticShadowDetectionDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_81_synthetic_shadow_detection.model import (
        ModelConfig,
        ShadowDetectionModel,
        shadow_detection_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        shadow_strength_min=0.25,
        shadow_strength_max=0.55,
    )
    ds = SyntheticShadowDetectionDataset(cfg)
    shadowed, targets = ds[0]

    assert tuple(shadowed.shape) == (3, 32, 32)
    assert set(targets.keys()) == {"shadow_mask", "boundary", "lit_image"}
    assert tuple(targets["shadow_mask"].shape) == (1, 32, 32)
    assert tuple(targets["boundary"].shape) == (1, 32, 32)
    assert tuple(targets["lit_image"].shape) == (3, 32, 32)
    assert shadowed.dtype == torch.float32
    assert targets["shadow_mask"].dtype == torch.float32
    assert targets["boundary"].dtype == torch.float32
    assert targets["lit_image"].dtype == torch.float32
    assert 0.0 <= float(shadowed.min().item()) <= float(shadowed.max().item()) <= 1.0
    assert 0.0 <= float(targets["shadow_mask"].min().item()) <= float(targets["shadow_mask"].max().item()) <= 1.0

    train_loader, _ = get_dataloaders(cfg)
    batch_shadowed, batch_targets = next(iter(train_loader))
    assert tuple(batch_shadowed.shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["shadow_mask"].shape) == (4, 1, 32, 32)
    assert tuple(batch_targets["boundary"].shape) == (4, 1, 32, 32)
    assert tuple(batch_targets["lit_image"].shape) == (4, 3, 32, 32)

    model = ShadowDetectionModel(ModelConfig(in_channels=3, hidden_channels=24, backbone_variant="context_shadow_tiny"))
    outputs = model(batch_shadowed)
    assert set(outputs.keys()) == {"logits", "shadow_mask", "boundary", "lit_image"}
    assert tuple(outputs["logits"].shape) == (4, 1, 32, 32)
    assert tuple(outputs["shadow_mask"].shape) == (4, 1, 32, 32)
    assert tuple(outputs["boundary"].shape) == (4, 1, 32, 32)
    assert tuple(outputs["lit_image"].shape) == (4, 3, 32, 32)

    loss, parts = shadow_detection_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"mask_loss", "boundary_loss", "lit_image_loss"}
    assert float(parts["mask_loss"]) >= 0.0
    assert float(parts["boundary_loss"]) >= 0.0
    assert float(parts["lit_image_loss"]) >= 0.0
    loss.backward()


def test_vision_shadow_detection_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_81_synthetic_shadow_detection.data import DataConfig
    from tracks.vision.lesson_81_synthetic_shadow_detection.model import ModelConfig
    from tracks.vision.lesson_81_synthetic_shadow_detection.train import (
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
            run_name="pytest_shadow_detection_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=11,
            num_workers=0,
            in_channels=3,
            shadow_strength_min=0.25,
            shadow_strength_max=0.55,
        ),
        ModelConfig(in_channels=3, hidden_channels=24, backbone_variant="context_shadow_tiny"),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_81_synthetic_shadow_detection" / "pytest_shadow_detection_smoke"
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
        "train_mask_loss",
        "train_boundary_loss",
        "train_lit_image_loss",
        "train_psnr",
        "eval_loss",
        "eval_mask_loss",
        "eval_boundary_loss",
        "eval_lit_image_loss",
        "eval_psnr",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
