"""PointCloud track smoke checks (torch-only)."""


def check_zoo_compact_classification() -> None:
    # 4.3) PointCloud lesson: local zoo (torch-only, synthetic data).
    from dlhub.paths import build_run_paths
    from tracks.pointcloud.lesson_04_pointcloud_zoo_compact_classification.data import (
        DataConfig as PCData,
    )
    from tracks.pointcloud.lesson_04_pointcloud_zoo_compact_classification.train import (
        TrainConfig as PCTrain,
    )
    from tracks.pointcloud.lesson_04_pointcloud_zoo_compact_classification.train import (
        run_training as run_pc,
    )

    run_pc(
        PCTrain(
            epochs=1,
            learning_rate=1e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            arch="pc:pointnet",
            width_mult=0.5,
            dropout=0.1,
        ),
        PCData(
            num_samples=256,
            num_points=64,
            batch_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    pc_paths = build_run_paths(
        track="pointcloud",
        lesson="lesson_04_pointcloud_zoo_compact_classification",
        run_name="smoke",
    )
    assert (pc_paths.run_dir / "config.json").is_file()
    assert (pc_paths.run_dir / "metrics.jsonl").is_file()
    assert (pc_paths.checkpoints_dir / "checkpoint.pt").is_file()


def check_pointnet_compact_classification() -> None:
    # 4.13) PointCloud lesson: compact PointNet classification (torch-only).
    from dlhub.paths import build_run_paths
    from tracks.pointcloud.lesson_01_pointnet_compact_classification.data import (
        DataConfig as PcData,
    )
    from tracks.pointcloud.lesson_01_pointnet_compact_classification.train import (
        TrainConfig as PcTrain,
    )
    from tracks.pointcloud.lesson_01_pointnet_compact_classification.train import (
        run_training as run_pc,
    )

    run_pc(
        PcTrain(
            epochs=1,
            learning_rate=1e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            hidden_features=32,
            dropout=0.0,
        ),
        PcData(
            num_samples=256,
            num_points=64,
            batch_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    pc_paths = build_run_paths(
        track="pointcloud", lesson="lesson_01_pointnet_compact_classification", run_name="smoke"
    )
    assert (pc_paths.run_dir / "config.json").is_file()
    assert (pc_paths.run_dir / "metrics.jsonl").is_file()
    assert (pc_paths.checkpoints_dir / "checkpoint.pt").is_file()


def check_dgcnn_compact_classification() -> None:
    # 4.14) PointCloud lesson: compact DGCNN classification (torch-only).
    from dlhub.paths import build_run_paths
    from tracks.pointcloud.lesson_02_dgcnn_compact_classification.data import DataConfig as DgData
    from tracks.pointcloud.lesson_02_dgcnn_compact_classification.train import (
        TrainConfig as DgTrain,
    )
    from tracks.pointcloud.lesson_02_dgcnn_compact_classification.train import (
        run_training as run_dg,
    )

    run_dg(
        DgTrain(
            epochs=1,
            learning_rate=1e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            k=5,
            hidden_features=32,
            dropout=0.0,
            dynamic_graph=True,
        ),
        DgData(
            num_samples=256,
            num_points=64,
            batch_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    dg_paths = build_run_paths(
        track="pointcloud", lesson="lesson_02_dgcnn_compact_classification", run_name="smoke"
    )
    assert (dg_paths.run_dir / "config.json").is_file()
    assert (dg_paths.run_dir / "metrics.jsonl").is_file()
    assert (dg_paths.checkpoints_dir / "checkpoint.pt").is_file()


def check_pointnet2_compact_classification() -> None:
    # 4.15) PointCloud lesson: compact PointNet2 classification (torch-only).
    from dlhub.paths import build_run_paths
    from tracks.pointcloud.lesson_03_pointnet2_compact_classification.data import (
        DataConfig as P2Data,
    )
    from tracks.pointcloud.lesson_03_pointnet2_compact_classification.train import (
        TrainConfig as P2Train,
    )
    from tracks.pointcloud.lesson_03_pointnet2_compact_classification.train import (
        run_training as run_p2,
    )

    run_p2(
        P2Train(
            epochs=1,
            learning_rate=1e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            npoint1=16,
            k1=8,
            npoint2=4,
            k2=4,
            hidden_features=32,
            dropout=0.0,
        ),
        P2Data(
            num_samples=256,
            num_points=64,
            batch_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
        ),
    )

    p2_paths = build_run_paths(
        track="pointcloud", lesson="lesson_03_pointnet2_compact_classification", run_name="smoke"
    )
    assert (p2_paths.run_dir / "config.json").is_file()
    assert (p2_paths.run_dir / "metrics.jsonl").is_file()
    assert (p2_paths.checkpoints_dir / "checkpoint.pt").is_file()
