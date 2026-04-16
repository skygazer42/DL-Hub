import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_image_stitching_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_63_synthetic_image_stitching.data import (
        DataConfig,
        SyntheticImageStitchingDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_63_synthetic_image_stitching.model import (
        ModelConfig,
        StitchingModel,
        stitching_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=48,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        in_channels=3,
        overlap_width=12,
    )

    dataset = SyntheticImageStitchingDataset(cfg)
    pair, target = dataset[0]
    assert set(pair.keys()) == {"left_view", "right_view"}
    assert tuple(pair["left_view"].shape) == (3, 48, 48)
    assert tuple(pair["right_view"].shape) == (3, 48, 48)
    assert tuple(target.shape) == (3, 48, 48)
    assert pair["left_view"].dtype == torch.float32
    assert pair["right_view"].dtype == torch.float32
    assert target.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    inputs, labels = next(iter(train_loader))
    assert set(inputs.keys()) == {"left_view", "right_view"}
    assert tuple(inputs["left_view"].shape) == (4, 3, 48, 48)
    assert tuple(inputs["right_view"].shape) == (4, 3, 48, 48)
    assert tuple(labels.shape) == (4, 3, 48, 48)

    model = StitchingModel(ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3))
    stitched = model(inputs)
    assert tuple(stitched.shape) == (4, 3, 48, 48)
    assert 0.0 <= float(stitched.min().item()) <= float(stitched.max().item()) <= 1.0

    loss, parts = stitching_loss(stitched, labels)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"reconstruction_loss", "seam_loss"}
    assert float(parts["reconstruction_loss"]) >= 0.0
    assert float(parts["seam_loss"]) >= 0.0
    loss.backward()


def test_image_stitching_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_63_synthetic_image_stitching.data import DataConfig
    from tracks.vision.lesson_63_synthetic_image_stitching.model import ModelConfig
    from tracks.vision.lesson_63_synthetic_image_stitching.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=63,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_stitching_smoke",
        ),
        DataConfig(
            num_samples=40,
            batch_size=4,
            image_size=48,
            val_fraction=0.2,
            seed=7,
            num_workers=0,
            in_channels=3,
            overlap_width=12,
        ),
        ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_63_synthetic_image_stitching" / "pytest_stitching_smoke"
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
        "train_seam_loss",
        "eval_loss",
        "eval_reconstruction_loss",
        "eval_seam_loss",
        "eval_psnr",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
