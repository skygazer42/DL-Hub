from __future__ import annotations

import importlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    ("module_name", "lesson", "train_kwargs", "data_kwargs"),
    [
        (
            "tracks.pointcloud.lesson_21_pointcloud_selfsupervised_msn.train",
            "lesson_21_pointcloud_selfsupervised_msn",
            {"arch": "msn_pointmae:msn_pointmae_tiny", "out_dim": 32},
            {"num_points": 32},
        ),
        (
            "tracks.pointcloud.lesson_22_pointcloud_selfsupervised_data2vec.train",
            "lesson_22_pointcloud_selfsupervised_data2vec",
            {
                "arch": "data2vec_pointmae:data2vec_pointmae_tiny",
                "predictor_hidden": 32,
            },
            {"num_points": 32},
        ),
        (
            "tracks.pointcloud.lesson_23_pointcloud_selfsupervised_ressl.train",
            "lesson_23_pointcloud_selfsupervised_ressl",
            {"arch": "ressl_pointnet:ressl_pointnet_tiny", "queue_size": 32},
            {"num_points": 32},
        ),
        (
            "tracks.vision.lesson_12_synthetic_detection_yolo.train",
            "lesson_12_synthetic_detection_yolo",
            {},
            {"image_size": 32, "stride": 4, "min_box_size": 6, "max_box_size": 12},
        ),
    ],
)
def test_epoch_and_stable_checkpoints_are_both_safe_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    lesson: str,
    train_kwargs: dict[str, object],
    data_kwargs: dict[str, object],
) -> None:
    train_module = importlib.import_module(module_name)
    data_module = importlib.import_module(module_name.rsplit(".", 1)[0] + ".data")
    run_name = "checkpoint-contract"
    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    train_cfg = train_module.TrainConfig(
        epochs=1,
        seed=0,
        device="cpu",
        max_train_batches=1,
        max_eval_batches=1,
        run_name=run_name,
        **train_kwargs,
    )
    data_cfg = data_module.DataConfig(
        num_samples=16,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        **data_kwargs,
    )

    assert train_module.run_training(train_cfg, data_cfg) == 0

    track = "vision" if ".vision." in module_name else "pointcloud"
    checkpoint_dir = tmp_path / track / lesson / run_name / "checkpoints"
    epoch_path = checkpoint_dir / "epoch_001.pt"
    stable_path = checkpoint_dir / "checkpoint.pt"
    for path in (epoch_path, stable_path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["epoch"] == 1
        assert isinstance(payload["model_state"], dict) and payload["model_state"]
        assert isinstance(payload["optimizer_state"], dict) and payload["optimizer_state"]
        assert isinstance(payload["extra"], dict)
    assert not list(checkpoint_dir.glob("*.tmp"))
