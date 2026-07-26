"""Vision track smoke checks (torch-only)."""


def check_mnist_lenet() -> None:
    # 4.2) Vision lesson (requires torchvision, no downloads in `fake` mode).
    try:
        import torchvision  # noqa: F401
    except Exception as exc:
        print("smoke_check: torchvision not available; skipping vision lesson.")
        print(f"- reason: {exc}")
    else:
        from dlhub.paths import build_run_paths
        from tracks.vision.lesson_01_mnist_lenet.data import DataConfig
        from tracks.vision.lesson_01_mnist_lenet.train import TrainConfig, run_training

        run_training(
            TrainConfig(
                epochs=1,
                learning_rate=1e-3,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
            ),
            DataConfig(dataset="fake", batch_size=32, num_workers=0),
        )

        vis_paths = build_run_paths(
            track="vision", lesson="lesson_01_mnist_lenet", run_name="smoke"
        )
        assert (vis_paths.run_dir / "config.json").is_file()
        assert (vis_paths.run_dir / "metrics.jsonl").is_file()
        assert (vis_paths.checkpoints_dir / "checkpoint.pt").is_file()


def check_synthetic_detection() -> None:
    # 4.2b) Vision lesson: synthetic detection (torch-only).
    from dlhub.paths import build_run_paths
    from tracks.vision.lesson_04_synthetic_detection_fcos.data import DataConfig as DetData
    from tracks.vision.lesson_04_synthetic_detection_fcos.train import TrainConfig as DetTrain
    from tracks.vision.lesson_04_synthetic_detection_fcos.train import run_training as run_det

    run_det(
        DetTrain(
            epochs=1,
            learning_rate=2e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            cls_pos_weight=30.0,
            reg_weight=2.0,
        ),
        DetData(
            num_samples=256,
            batch_size=32,
            image_size=64,
            stride=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.15,
            min_box_size=10,
            max_box_size=28,
        ),
    )

    det_paths = build_run_paths(
        track="vision", lesson="lesson_04_synthetic_detection_fcos", run_name="smoke"
    )
    assert (det_paths.run_dir / "config.json").is_file()
    assert (det_paths.run_dir / "metrics.jsonl").is_file()
    assert (det_paths.checkpoints_dir / "checkpoint.pt").is_file()


def check_vit_toy_classification() -> None:
    # 4.2c) Vision lesson: ViT toy classification (torch-only).
    from dlhub.paths import build_run_paths
    from tracks.vision.lesson_05_vit_toy_classification.data import DataConfig as VitData
    from tracks.vision.lesson_05_vit_toy_classification.train import TrainConfig as VitTrain
    from tracks.vision.lesson_05_vit_toy_classification.train import run_training as run_vit

    run_vit(
        VitTrain(
            epochs=1,
            learning_rate=2e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            patch_size=8,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
            dropout=0.1,
        ),
        VitData(
            num_samples=256,
            batch_size=32,
            image_size=64,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.15,
            min_square=8,
            max_square=20,
            num_classes=4,
        ),
    )

    vit_paths = build_run_paths(
        track="vision", lesson="lesson_05_vit_toy_classification", run_name="smoke"
    )
    assert (vit_paths.run_dir / "config.json").is_file()
    assert (vit_paths.run_dir / "metrics.jsonl").is_file()
    assert (vit_paths.checkpoints_dir / "checkpoint.pt").is_file()


def check_swin_toy_classification() -> None:
    # 4.2d) Vision lesson: Swin-style toy classification (torch-only).
    from dlhub.paths import build_run_paths
    from tracks.vision.lesson_06_swin_toy_classification.data import DataConfig as SwinData
    from tracks.vision.lesson_06_swin_toy_classification.train import TrainConfig as SwinTrain
    from tracks.vision.lesson_06_swin_toy_classification.train import run_training as run_swin

    run_swin(
        SwinTrain(
            epochs=1,
            learning_rate=2e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            patch_size=4,
            embed_dim=64,
            num_heads=4,
            depth=2,
            window_size=4,
            mlp_ratio=2.0,
            dropout=0.1,
        ),
        SwinData(
            num_samples=256,
            batch_size=32,
            image_size=64,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.15,
            min_square=8,
            max_square=20,
            num_classes=4,
        ),
    )

    swin_paths = build_run_paths(
        track="vision", lesson="lesson_06_swin_toy_classification", run_name="smoke"
    )
    assert (swin_paths.run_dir / "config.json").is_file()
    assert (swin_paths.run_dir / "metrics.jsonl").is_file()
    assert (swin_paths.checkpoints_dir / "checkpoint.pt").is_file()


def check_toy_keypoint_regression() -> None:
    # 4.2e) Vision lesson: toy keypoint regression (torch-only).
    from dlhub.paths import build_run_paths
    from tracks.vision.lesson_07_toy_keypoint_regression.data import DataConfig as KptData
    from tracks.vision.lesson_07_toy_keypoint_regression.train import TrainConfig as KptTrain
    from tracks.vision.lesson_07_toy_keypoint_regression.train import run_training as run_kpt

    run_kpt(
        KptTrain(
            epochs=1,
            learning_rate=2e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            hidden_channels=16,
            num_blocks=2,
            dropout=0.0,
        ),
        KptData(
            num_samples=256,
            batch_size=32,
            image_size=64,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.10,
            dot_sigma=1.5,
        ),
    )

    kpt_paths = build_run_paths(
        track="vision", lesson="lesson_07_toy_keypoint_regression", run_name="smoke"
    )
    assert (kpt_paths.run_dir / "config.json").is_file()
    assert (kpt_paths.run_dir / "metrics.jsonl").is_file()
    assert (kpt_paths.checkpoints_dir / "checkpoint.pt").is_file()


def check_synthetic_segmentation() -> None:
    # 4.2f) Vision lesson: synthetic segmentation (torch-only).
    from dlhub.paths import build_run_paths
    from tracks.vision.lesson_08_synthetic_segmentation_unet.data import DataConfig as SegData
    from tracks.vision.lesson_08_synthetic_segmentation_unet.train import (
        TrainConfig as SegTrain,
    )
    from tracks.vision.lesson_08_synthetic_segmentation_unet.train import (
        run_training as run_seg,
    )

    run_seg(
        SegTrain(
            epochs=1,
            learning_rate=2e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            base_channels=16,
            dropout=0.0,
            threshold=0.5,
        ),
        SegData(
            num_samples=256,
            batch_size=8,
            image_size=64,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.15,
            min_rect=10,
            max_rect=28,
        ),
    )

    seg_paths = build_run_paths(
        track="vision", lesson="lesson_08_synthetic_segmentation_unet", run_name="smoke"
    )
    assert (seg_paths.run_dir / "config.json").is_file()
    assert (seg_paths.run_dir / "metrics.jsonl").is_file()
    assert (seg_paths.checkpoints_dir / "checkpoint.pt").is_file()


def check_cnn_backbones() -> None:
    # 4.2g) Vision lesson: classic CNN backbones (torch-only).
    from dlhub.paths import build_run_paths
    from tracks.vision.lesson_09_cnn_backbones_toy_classification.data import (
        DataConfig as CnnData,
    )
    from tracks.vision.lesson_09_cnn_backbones_toy_classification.train import (
        TrainConfig as CnnTrain,
    )
    from tracks.vision.lesson_09_cnn_backbones_toy_classification.train import (
        run_training as run_cnn,
    )

    run_cnn(
        CnnTrain(
            epochs=1,
            learning_rate=2e-3,
            seed=0,
            device="cpu",
            max_train_batches=1,
            max_eval_batches=1,
            run_name="smoke",
            arch="resnet18",
            width_mult=0.5,
            dropout=0.0,
        ),
        CnnData(
            num_samples=256,
            batch_size=32,
            image_size=64,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.15,
            min_square=8,
            max_square=20,
            num_classes=4,
        ),
    )

    cnn_paths = build_run_paths(
        track="vision", lesson="lesson_09_cnn_backbones_toy_classification", run_name="smoke"
    )
    assert (cnn_paths.run_dir / "config.json").is_file()
    assert (cnn_paths.run_dir / "metrics.jsonl").is_file()
    assert (cnn_paths.checkpoints_dir / "checkpoint.pt").is_file()
