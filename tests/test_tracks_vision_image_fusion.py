import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_image_fusion_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_25_synthetic_image_fusion.data import (
        DataConfig,
        SyntheticImageFusionDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_25_synthetic_image_fusion.model import (
        FusionModel,
        ModelConfig,
        fusion_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        blur_kernel_size=7,
        blur_sigma=1.2,
    )

    ds = SyntheticImageFusionDataset(cfg)
    pair, target = ds[0]

    assert set(pair.keys()) == {"near_focus", "far_focus"}
    assert tuple(pair["near_focus"].shape) == (3, 32, 32)
    assert tuple(pair["far_focus"].shape) == (3, 32, 32)
    assert tuple(target.shape) == (3, 32, 32)
    assert pair["near_focus"].dtype == torch.float32
    assert pair["far_focus"].dtype == torch.float32
    assert target.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    inputs, labels = next(iter(train_loader))
    assert set(inputs.keys()) == {"near_focus", "far_focus"}
    assert tuple(inputs["near_focus"].shape) == (4, 3, 32, 32)
    assert tuple(inputs["far_focus"].shape) == (4, 3, 32, 32)
    assert tuple(labels.shape) == (4, 3, 32, 32)

    model = FusionModel(ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3))
    fused = model(inputs)
    assert tuple(fused.shape) == (4, 3, 32, 32)
    assert 0.0 <= float(fused.min().item()) <= float(fused.max().item()) <= 1.0

    loss, parts = fusion_loss(fused, labels)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"reconstruction_loss", "consistency_loss"}
    assert float(parts["reconstruction_loss"]) >= 0.0
    assert float(parts["consistency_loss"]) >= 0.0
    loss.backward()


def test_vision_image_fusion_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_25_synthetic_image_fusion.data import DataConfig
    from tracks.vision.lesson_25_synthetic_image_fusion.model import ModelConfig
    from tracks.vision.lesson_25_synthetic_image_fusion.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=0,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_fusion_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=7,
            num_workers=0,
            in_channels=3,
            blur_kernel_size=7,
            blur_sigma=1.2,
        ),
        ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3),
    )
    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_25_synthetic_image_fusion" / "pytest_fusion_smoke"
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
        "train_reconstruction_loss",
        "train_consistency_loss",
        "eval_loss",
        "eval_reconstruction_loss",
        "eval_consistency_loss",
        "eval_psnr",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
