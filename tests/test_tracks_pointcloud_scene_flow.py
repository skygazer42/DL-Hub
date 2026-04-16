import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_scene_flow_batch_contract_and_loss_smoke() -> None:
    from tracks.pointcloud.lesson_25_toy_scene_flow_estimation.data import (
        DataConfig,
        ToySceneFlowDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_25_toy_scene_flow_estimation.model import (
        ModelConfig,
        ToySceneFlowEstimator,
        scene_flow_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        num_points=48,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        translation_scale=0.35,
    )

    ds = ToySceneFlowDataset(cfg)
    source, target, flow = ds[0]
    assert tuple(source.shape) == (48, 3)
    assert tuple(target.shape) == (48, 3)
    assert tuple(flow.shape) == (48, 3)
    assert source.dtype == torch.float32
    assert target.dtype == torch.float32
    assert flow.dtype == torch.float32
    assert torch.allclose(target - source, flow, atol=1e-6)

    train_loader, _ = get_dataloaders(cfg)
    source_batch, target_batch, flow_batch = next(iter(train_loader))
    assert tuple(source_batch.shape) == (4, 48, 3)
    assert tuple(target_batch.shape) == (4, 48, 3)
    assert tuple(flow_batch.shape) == (4, 48, 3)

    model = ToySceneFlowEstimator(ModelConfig(hidden_features=32))
    pred_flow = model(source_batch, target_batch)
    assert tuple(pred_flow.shape) == (4, 48, 3)

    loss, parts = scene_flow_loss(pred_flow, flow_batch)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"flow_loss", "endpoint_error"}
    assert float(parts["flow_loss"]) >= 0.0
    assert float(parts["endpoint_error"]) >= 0.0
    loss.backward()


def test_pointcloud_scene_flow_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_25_toy_scene_flow_estimation.data import DataConfig
    from tracks.pointcloud.lesson_25_toy_scene_flow_estimation.model import (
        ModelConfig,
    )
    from tracks.pointcloud.lesson_25_toy_scene_flow_estimation.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=5e-2,
            seed=0,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_scene_flow_smoke",
        ),
        DataConfig(
            num_samples=48,
            num_points=48,
            batch_size=4,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
            translation_scale=0.35,
        ),
        ModelConfig(hidden_features=32),
    )

    assert exit_code == 0

    run_dir = tmp_path / "pointcloud" / "lesson_25_toy_scene_flow_estimation" / "pytest_scene_flow_smoke"
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
    for key in ("train_loss", "train_epe", "eval_loss", "eval_epe"):
        assert key in record
        assert float(record[key]) >= 0.0
