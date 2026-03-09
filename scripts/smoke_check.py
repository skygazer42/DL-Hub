
import sys
from pathlib import Path

import numpy as np


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def main() -> int:
    _ensure_repo_root_on_path()

    from ml_algorithms.python.kmeans import KMeans
    from ml_algorithms.python.linear_models import LogisticRegression
    from optimization.python.losses import mean_squared_error
    from optimization.python.lr_schedulers import WarmupCosine
    from optimization.python.metrics import accuracy_score
    from optimization.python.optimizers import Adam

    rng = np.random.default_rng(0)

    # 1) A tiny linear classification sanity check.
    x = rng.normal(size=(256, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    clf = LogisticRegression(learning_rate=0.1, epochs=500).fit(x, y)
    preds = clf.predict(x)
    acc = accuracy_score(y, preds)
    assert acc > 0.85

    # 2) A tiny clustering sanity check.
    kmeans = KMeans(n_clusters=3, random_state=0).fit(x)
    assert kmeans.cluster_centers_.shape == (3, 2)
    assert kmeans.labels_.shape == (x.shape[0],)

    # 3) Optimizer + scheduler plumbing sanity check.
    params = {"w": rng.normal(size=(3, 3)), "b": np.zeros(3)}
    grads = {"w": np.ones((3, 3)) * 0.1, "b": np.ones(3) * 0.01}
    opt = Adam(learning_rate=1e-3)
    scheduler = WarmupCosine(base_lr=1e-3, warmup_steps=2, max_steps=10)

    for _ in range(3):
        opt.learning_rate = scheduler.step()
        params = opt.step(params, grads)

    mse = mean_squared_error(np.zeros_like(params["b"]), params["b"])
    assert mse >= 0.0

    # 4) PyTorch tracks sanity checks.
    ran_pytorch = False
    try:
        import torch  # noqa: F401
    except Exception as exc:
        print("smoke_check: torch not available; skipping PyTorch lessons.")
        print(f"- reason: {exc}")
    else:
        ran_pytorch = True

        # 4.1) Foundations lesson (no downloads, torch-only).
        from dlhub.paths import build_run_paths
        from tracks.foundations.lesson_02_linear_regression_autograd.data import DataConfig as RegData
        from tracks.foundations.lesson_02_linear_regression_autograd.train import (
            TrainConfig as RegTrain,
            run_training as run_regression,
        )

        run_regression(
            RegTrain(
                epochs=1,
                learning_rate=0.1,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
            ),
            RegData(num_samples=128, batch_size=64, noise_std=0.1),
        )

        reg_paths = build_run_paths(
            track="foundations", lesson="lesson_02_linear_regression_autograd", run_name="smoke"
        )
        assert (reg_paths.run_dir / "config.json").is_file()
        assert (reg_paths.run_dir / "metrics.jsonl").is_file()
        assert (reg_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.2) Vision lesson (requires torchvision, no downloads in `fake` mode).
        try:
            import torchvision  # noqa: F401
        except Exception as exc:
            print("smoke_check: torchvision not available; skipping vision lesson.")
            print(f"- reason: {exc}")
        else:
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

        # 4.2b) Vision lesson: synthetic detection (torch-only).
        from tracks.vision.lesson_04_synthetic_detection_fcos.data import DataConfig as DetData
        from tracks.vision.lesson_04_synthetic_detection_fcos.train import (
            TrainConfig as DetTrain,
            run_training as run_det,
        )

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

        det_paths = build_run_paths(track="vision", lesson="lesson_04_synthetic_detection_fcos", run_name="smoke")
        assert (det_paths.run_dir / "config.json").is_file()
        assert (det_paths.run_dir / "metrics.jsonl").is_file()
        assert (det_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.2c) Vision lesson: ViT toy classification (torch-only).
        from tracks.vision.lesson_05_vit_toy_classification.data import DataConfig as VitData
        from tracks.vision.lesson_05_vit_toy_classification.train import (
            TrainConfig as VitTrain,
            run_training as run_vit,
        )

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

        vit_paths = build_run_paths(track="vision", lesson="lesson_05_vit_toy_classification", run_name="smoke")
        assert (vit_paths.run_dir / "config.json").is_file()
        assert (vit_paths.run_dir / "metrics.jsonl").is_file()
        assert (vit_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.2d) Vision lesson: Swin-style toy classification (torch-only).
        from tracks.vision.lesson_06_swin_toy_classification.data import DataConfig as SwinData
        from tracks.vision.lesson_06_swin_toy_classification.train import (
            TrainConfig as SwinTrain,
            run_training as run_swin,
        )

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

        swin_paths = build_run_paths(track="vision", lesson="lesson_06_swin_toy_classification", run_name="smoke")
        assert (swin_paths.run_dir / "config.json").is_file()
        assert (swin_paths.run_dir / "metrics.jsonl").is_file()
        assert (swin_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.3) PointCloud lesson: local zoo (torch-only, synthetic data).
        from tracks.pointcloud.lesson_04_pointcloud_zoo_toy_classification.data import DataConfig as PCData
        from tracks.pointcloud.lesson_04_pointcloud_zoo_toy_classification.train import (
            TrainConfig as PCTrain,
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

        pc_paths = build_run_paths(track="pointcloud", lesson="lesson_04_pointcloud_zoo_toy_classification", run_name="smoke")
        assert (pc_paths.run_dir / "config.json").is_file()
        assert (pc_paths.run_dir / "metrics.jsonl").is_file()
        assert (pc_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.2e) Vision lesson: toy keypoint regression (torch-only).
        from tracks.vision.lesson_07_toy_keypoint_regression.data import DataConfig as KptData
        from tracks.vision.lesson_07_toy_keypoint_regression.train import (
            TrainConfig as KptTrain,
            run_training as run_kpt,
        )

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

        kpt_paths = build_run_paths(track="vision", lesson="lesson_07_toy_keypoint_regression", run_name="smoke")
        assert (kpt_paths.run_dir / "config.json").is_file()
        assert (kpt_paths.run_dir / "metrics.jsonl").is_file()
        assert (kpt_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.2f) Vision lesson: synthetic segmentation (torch-only).
        from tracks.vision.lesson_08_synthetic_segmentation_unet.data import DataConfig as SegData
        from tracks.vision.lesson_08_synthetic_segmentation_unet.train import (
            TrainConfig as SegTrain,
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

        seg_paths = build_run_paths(track="vision", lesson="lesson_08_synthetic_segmentation_unet", run_name="smoke")
        assert (seg_paths.run_dir / "config.json").is_file()
        assert (seg_paths.run_dir / "metrics.jsonl").is_file()
        assert (seg_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.2g) Vision lesson: classic CNN backbones (torch-only).
        from tracks.vision.lesson_09_cnn_backbones_toy_classification.data import DataConfig as CnnData
        from tracks.vision.lesson_09_cnn_backbones_toy_classification.train import (
            TrainConfig as CnnTrain,
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

        cnn_paths = build_run_paths(track="vision", lesson="lesson_09_cnn_backbones_toy_classification", run_name="smoke")
        assert (cnn_paths.run_dir / "config.json").is_file()
        assert (cnn_paths.run_dir / "metrics.jsonl").is_file()
        assert (cnn_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.3) LLM lesson: toy causal LM (torch-only).
        from tracks.llm.lesson_01_toy_causal_lm_transformer.data import DataConfig as LmData
        from tracks.llm.lesson_01_toy_causal_lm_transformer.train import (
            TrainConfig as LmTrain,
            run_training as run_lm,
        )

        run_lm(
            LmTrain(
                epochs=1,
                learning_rate=2e-3,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
                embed_dim=64,
                num_heads=4,
                num_layers=2,
                ff_dim=128,
                dropout=0.1,
            ),
            LmData(num_samples=256, batch_size=32, seq_length=32, base_vocab_size=64, val_fraction=0.2, seed=0, num_workers=0),
        )

        lm_paths = build_run_paths(track="llm", lesson="lesson_01_toy_causal_lm_transformer", run_name="smoke")
        assert (lm_paths.run_dir / "config.json").is_file()
        assert (lm_paths.run_dir / "metrics.jsonl").is_file()
        assert (lm_paths.run_dir / "vocab.json").is_file()
        assert (lm_paths.run_dir / "samples.jsonl").is_file()
        assert (lm_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.3) GNN lesson (torch-only, fully synthetic).
        from tracks.gnn.lesson_01_toy_graph_classification.data import DataConfig as GnnData
        from tracks.gnn.lesson_01_toy_graph_classification.train import (
            TrainConfig as GnnTrain,
            run_training as run_gnn,
        )

        run_gnn(
            GnnTrain(
                epochs=1,
                learning_rate=1e-2,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
                hidden_features=16,
            ),
            GnnData(num_graphs=64, num_nodes=10, batch_size=16, val_fraction=0.2, seed=0, num_workers=0),
        )

        gnn_paths = build_run_paths(
            track="gnn", lesson="lesson_01_toy_graph_classification", run_name="smoke"
        )
        assert (gnn_paths.run_dir / "config.json").is_file()
        assert (gnn_paths.run_dir / "metrics.jsonl").is_file()
        assert (gnn_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.4) GNN lesson: GIN (toy, torch-only).
        from tracks.gnn.lesson_02_gin_toy_graph_classification.data import DataConfig as GinData
        from tracks.gnn.lesson_02_gin_toy_graph_classification.train import (
            TrainConfig as GinTrain,
            run_training as run_gin,
        )

        run_gin(
            GinTrain(
                epochs=1,
                learning_rate=1e-2,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
                hidden_features=16,
                num_layers=3,
                num_mlp_layers=2,
                neighbor_pooling="sum",
                graph_pooling="mean",
                learn_eps=False,
                dropout=0.0,
            ),
            GinData(num_graphs=64, num_nodes=10, batch_size=16, val_fraction=0.2, seed=0, num_workers=0),
        )

        gin_paths = build_run_paths(
            track="gnn", lesson="lesson_02_gin_toy_graph_classification", run_name="smoke"
        )
        assert (gin_paths.run_dir / "config.json").is_file()
        assert (gin_paths.run_dir / "metrics.jsonl").is_file()
        assert (gin_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.5) GNN lesson: GAT (toy, torch-only).
        from tracks.gnn.lesson_03_gat_toy_graph_classification.data import DataConfig as GatData
        from tracks.gnn.lesson_03_gat_toy_graph_classification.train import (
            TrainConfig as GatTrain,
            run_training as run_gat,
        )

        run_gat(
            GatTrain(
                epochs=1,
                learning_rate=1e-2,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
                hidden_features=16,
                num_heads=4,
                dropout=0.1,
                alpha=0.2,
            ),
            GatData(num_graphs=64, num_nodes=10, batch_size=16, val_fraction=0.2, seed=0, num_workers=0),
        )

        gat_paths = build_run_paths(
            track="gnn", lesson="lesson_03_gat_toy_graph_classification", run_name="smoke"
        )
        assert (gat_paths.run_dir / "config.json").is_file()
        assert (gat_paths.run_dir / "metrics.jsonl").is_file()
        assert (gat_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.6) GNN lesson: Cora node classification (GCN).
        from tracks.gnn.lesson_04_cora_node_classification_gcn.train import (
            TrainConfig as CoraTrain,
            run_training as run_cora,
        )

        run_cora(
            CoraTrain(
                epochs=1,
                learning_rate=1e-2,
                weight_decay=5e-4,
                seed=0,
                device="cpu",
                run_name="smoke",
                hidden_features=16,
                dropout=0.5,
            )
        )

        cora_paths = build_run_paths(
            track="gnn", lesson="lesson_04_cora_node_classification_gcn", run_name="smoke"
        )
        assert (cora_paths.run_dir / "config.json").is_file()
        assert (cora_paths.run_dir / "metrics.jsonl").is_file()
        assert (cora_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.7) GNN lesson: Cora label propagation baseline (no learning).
        from tracks.gnn.lesson_05_label_propagation_cora.train import (
            TrainConfig as LpTrain,
            run_training as run_lp,
        )

        run_lp(
            LpTrain(
                num_layers=3,
                alpha=0.9,
                clamp_labeled=True,
                seed=0,
                device="cpu",
                run_name="smoke",
            )
        )

        lp_paths = build_run_paths(track="gnn", lesson="lesson_05_label_propagation_cora", run_name="smoke")
        assert (lp_paths.run_dir / "config.json").is_file()
        assert (lp_paths.run_dir / "metrics.jsonl").is_file()
        assert (lp_paths.run_dir / "preds.pt").is_file()
        assert (lp_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.8) GNN lesson: Cora node classification (GraphSAGE, full-batch).
        from tracks.gnn.lesson_06_graphsage_cora.train import (
            TrainConfig as SageTrain,
            run_training as run_sage,
        )

        run_sage(
            SageTrain(
                epochs=1,
                learning_rate=1e-2,
                weight_decay=5e-4,
                seed=0,
                device="cpu",
                run_name="smoke",
                hidden_features=16,
                dropout=0.0,
            )
        )

        sage_paths = build_run_paths(track="gnn", lesson="lesson_06_graphsage_cora", run_name="smoke")
        assert (sage_paths.run_dir / "config.json").is_file()
        assert (sage_paths.run_dir / "metrics.jsonl").is_file()
        assert (sage_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.9) GNN lesson: Karate SDNE embeddings (torch-only, tiny).
        from tracks.gnn.lesson_07_sdne_karate_embedding.train import (
            TrainConfig as SdneTrain,
            run_training as run_sdne,
        )

        run_sdne(
            SdneTrain(
                epochs=3,
                learning_rate=1e-3,
                lambda_smooth=1.0,
                seed=0,
                device="cpu",
                run_name="smoke",
                embed_dim=8,
                hidden_dim=32,
                dropout=0.0,
            )
        )

        sdne_paths = build_run_paths(track="gnn", lesson="lesson_07_sdne_karate_embedding", run_name="smoke")
        assert (sdne_paths.run_dir / "config.json").is_file()
        assert (sdne_paths.run_dir / "metrics.jsonl").is_file()
        assert (sdne_paths.run_dir / "embeddings.pt").is_file()
        assert (sdne_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.10) GNN lesson: Karate LINE embeddings (torch-only, tiny).
        from tracks.gnn.lesson_08_line_karate_embedding.train import (
            TrainConfig as LineTrain,
            run_training as run_line,
        )

        run_line(
            LineTrain(
                epochs=1,
                steps_per_epoch=10,
                batch_size=64,
                negative_samples=3,
                learning_rate=1e-2,
                seed=0,
                device="cpu",
                run_name="smoke",
                embed_dim=8,
                order=2,
            )
        )

        line_paths = build_run_paths(track="gnn", lesson="lesson_08_line_karate_embedding", run_name="smoke")
        assert (line_paths.run_dir / "config.json").is_file()
        assert (line_paths.run_dir / "metrics.jsonl").is_file()
        assert (line_paths.run_dir / "embeddings.pt").is_file()
        assert (line_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.11) NLP lesson (toy, torch-only).
        from tracks.nlp.lesson_01_toy_text_classification.data import DataConfig as NlpData
        from tracks.nlp.lesson_01_toy_text_classification.train import (
            TrainConfig as NlpTrain,
            run_training as run_nlp,
        )

        run_nlp(
            NlpTrain(
                epochs=1,
                learning_rate=1e-2,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
                embed_dim=32,
                dropout=0.1,
            ),
            NlpData(num_samples=256, batch_size=32, max_length=16, val_fraction=0.2, seed=0, num_workers=0),
        )

        nlp_paths = build_run_paths(track="nlp", lesson="lesson_01_toy_text_classification", run_name="smoke")
        assert (nlp_paths.run_dir / "config.json").is_file()
        assert (nlp_paths.run_dir / "metrics.jsonl").is_file()
        assert (nlp_paths.run_dir / "vocab.json").is_file()
        assert (nlp_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.12) NLP lesson: TextCNN (toy, torch-only).
        from tracks.nlp.lesson_05_toy_text_classification_textcnn.data import DataConfig as CnnData
        from tracks.nlp.lesson_05_toy_text_classification_textcnn.train import (
            TrainConfig as CnnTrain,
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
                embed_dim=32,
                dropout=0.2,
                num_filters=32,
            ),
            CnnData(num_samples=256, batch_size=32, max_length=16, val_fraction=0.2, seed=0, num_workers=0),
        )

        cnn_paths = build_run_paths(track="nlp", lesson="lesson_05_toy_text_classification_textcnn", run_name="smoke")
        assert (cnn_paths.run_dir / "config.json").is_file()
        assert (cnn_paths.run_dir / "metrics.jsonl").is_file()
        assert (cnn_paths.run_dir / "vocab.json").is_file()
        assert (cnn_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.13) NLP lesson: BiLSTM classifier (toy, torch-only).
        from tracks.nlp.lesson_06_toy_text_classification_bilstm.data import DataConfig as RnnData
        from tracks.nlp.lesson_06_toy_text_classification_bilstm.train import (
            TrainConfig as RnnTrain,
            run_training as run_rnn,
        )

        run_rnn(
            RnnTrain(
                epochs=1,
                learning_rate=2e-3,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
                embed_dim=32,
                hidden_dim=32,
                dropout=0.2,
            ),
            RnnData(num_samples=256, batch_size=32, max_length=16, val_fraction=0.2, seed=0, num_workers=0),
        )

        rnn_paths = build_run_paths(track="nlp", lesson="lesson_06_toy_text_classification_bilstm", run_name="smoke")
        assert (rnn_paths.run_dir / "config.json").is_file()
        assert (rnn_paths.run_dir / "metrics.jsonl").is_file()
        assert (rnn_paths.run_dir / "vocab.json").is_file()
        assert (rnn_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.10) NLP lesson: transformer encoder (toy, torch-only).
        from tracks.nlp.lesson_02_toy_text_classification_transformer.data import (
            DataConfig as NlpTrData,
        )
        from tracks.nlp.lesson_02_toy_text_classification_transformer.train import (
            TrainConfig as NlpTrTrain,
            run_training as run_nlp_tr,
        )

        run_nlp_tr(
            NlpTrTrain(
                epochs=1,
                learning_rate=3e-4,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
                embed_dim=32,
                num_heads=4,
                num_layers=2,
                ff_dim=64,
                dropout=0.1,
            ),
            NlpTrData(num_samples=256, batch_size=32, max_length=16, val_fraction=0.2, seed=0, num_workers=0),
        )

        nlp_tr_paths = build_run_paths(
            track="nlp", lesson="lesson_02_toy_text_classification_transformer", run_name="smoke"
        )
        assert (nlp_tr_paths.run_dir / "config.json").is_file()
        assert (nlp_tr_paths.run_dir / "metrics.jsonl").is_file()
        assert (nlp_tr_paths.run_dir / "vocab.json").is_file()
        assert (nlp_tr_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.11) NLP lesson: seq2seq + attention (toy, torch-only).
        from tracks.nlp.lesson_04_toy_seq2seq_attention_generation.data import DataConfig as S2SData
        from tracks.nlp.lesson_04_toy_seq2seq_attention_generation.train import (
            TrainConfig as S2STrain,
            run_training as run_s2s,
        )

        run_s2s(
            S2STrain(
                epochs=1,
                learning_rate=2e-3,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
                embed_dim=32,
                hidden_dim=64,
                dropout=0.1,
            ),
            S2SData(
                num_samples=256,
                batch_size=32,
                min_len=6,
                max_len=12,
                base_vocab_size=24,
                val_fraction=0.2,
                seed=0,
                num_workers=0,
            ),
        )

        s2s_paths = build_run_paths(track="nlp", lesson="lesson_04_toy_seq2seq_attention_generation", run_name="smoke")
        assert (s2s_paths.run_dir / "config.json").is_file()
        assert (s2s_paths.run_dir / "metrics.jsonl").is_file()
        assert (s2s_paths.run_dir / "vocab.json").is_file()
        assert (s2s_paths.run_dir / "samples.jsonl").is_file()
        assert (s2s_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.11) NLP lesson: toy NER (BiLSTM, torch-only).
        from tracks.nlp.lesson_03_toy_ner_bilstm.data import DataConfig as NerData
        from tracks.nlp.lesson_03_toy_ner_bilstm.train import (
            TrainConfig as NerTrain,
            run_training as run_ner,
        )

        run_ner(
            NerTrain(
                epochs=1,
                learning_rate=1e-3,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
                embed_dim=32,
                hidden_dim=64,
                dropout=0.1,
            ),
            NerData(num_samples=256, batch_size=32, max_length=16, val_fraction=0.2, seed=0, num_workers=0),
        )

        ner_paths = build_run_paths(track="nlp", lesson="lesson_03_toy_ner_bilstm", run_name="smoke")
        assert (ner_paths.run_dir / "config.json").is_file()
        assert (ner_paths.run_dir / "metrics.jsonl").is_file()
        assert (ner_paths.run_dir / "vocab.json").is_file()
        assert (ner_paths.run_dir / "tags.json").is_file()
        assert (ner_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.12) NLP lesson: toy reading comprehension (span prediction).
        from tracks.nlp.lesson_07_reading_comprehension.data import DataConfig as RcData
        from tracks.nlp.lesson_07_reading_comprehension.train import (
            TrainConfig as RcTrain,
            run_training as run_rc,
        )

        run_rc(
            RcTrain(
                epochs=1,
                learning_rate=3e-3,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
                embed_dim=32,
                hidden_dim=32,
                dropout=0.1,
            ),
            RcData(
                num_samples=256,
                batch_size=32,
                context_length=32,
                question_length=4,
                val_fraction=0.2,
                seed=0,
                num_workers=0,
            ),
        )

        rc_paths = build_run_paths(track="nlp", lesson="lesson_07_reading_comprehension", run_name="smoke")
        assert (rc_paths.run_dir / "config.json").is_file()
        assert (rc_paths.run_dir / "metrics.jsonl").is_file()
        assert (rc_paths.run_dir / "vocab.json").is_file()
        assert (rc_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.13) Generative lesson: VAE (fake, torch-only).
        from tracks.generative.lesson_01_vae_mnist.data import DataConfig as VaeData
        from tracks.generative.lesson_01_vae_mnist.model import ModelConfig as VaeModel
        from tracks.generative.lesson_01_vae_mnist.train import (
            TrainConfig as VaeTrain,
            run_training as run_vae,
        )

        run_vae(
            VaeTrain(
                epochs=1,
                learning_rate=1e-3,
                beta=1.0,
                seed=0,
                device="cpu",
                max_train_batches=1,
                max_eval_batches=1,
                run_name="smoke",
            ),
            VaeData(dataset="fake", batch_size=64, num_workers=0, num_samples=256, seed=0, val_fraction=0.2),
            VaeModel(latent_dim=8, hidden_dim=64),
        )

        vae_paths = build_run_paths(track="generative", lesson="lesson_01_vae_mnist", run_name="smoke")
        assert (vae_paths.run_dir / "config.json").is_file()
        assert (vae_paths.run_dir / "metrics.jsonl").is_file()
        assert (vae_paths.run_dir / "samples.pt").is_file()
        assert (vae_paths.run_dir / "recons.pt").is_file()
        assert (vae_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.14) Generative lesson: GAN (fake, torch-only).
        from tracks.generative.lesson_02_gan_mnist.data import DataConfig as GanData
        from tracks.generative.lesson_02_gan_mnist.model import ModelConfig as GanModel
        from tracks.generative.lesson_02_gan_mnist.train import (
            TrainConfig as GanTrain,
            run_training as run_gan,
        )

        run_gan(
            GanTrain(
                epochs=1,
                learning_rate=2e-4,
                beta1=0.5,
                beta2=0.999,
                seed=0,
                device="cpu",
                max_train_batches=1,
                run_name="smoke",
                label_smoothing=0.0,
            ),
            GanData(dataset="fake", batch_size=64, num_workers=0, num_samples=256, seed=0),
            GanModel(z_dim=16, hidden_dim=64),
        )

        gan_paths = build_run_paths(track="generative", lesson="lesson_02_gan_mnist", run_name="smoke")
        assert (gan_paths.run_dir / "config.json").is_file()
        assert (gan_paths.run_dir / "metrics.jsonl").is_file()
        assert (gan_paths.run_dir / "samples.pt").is_file()
        assert (gan_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.13) PointCloud lesson: toy PointNet classification (torch-only).
        from tracks.pointcloud.lesson_01_pointnet_toy_classification.data import DataConfig as PcData
        from tracks.pointcloud.lesson_01_pointnet_toy_classification.train import (
            TrainConfig as PcTrain,
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
            PcData(num_samples=256, num_points=64, batch_size=32, val_fraction=0.2, seed=0, num_workers=0),
        )

        pc_paths = build_run_paths(
            track="pointcloud", lesson="lesson_01_pointnet_toy_classification", run_name="smoke"
        )
        assert (pc_paths.run_dir / "config.json").is_file()
        assert (pc_paths.run_dir / "metrics.jsonl").is_file()
        assert (pc_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.14) PointCloud lesson: toy DGCNN classification (torch-only).
        from tracks.pointcloud.lesson_02_dgcnn_toy_classification.data import DataConfig as DgData
        from tracks.pointcloud.lesson_02_dgcnn_toy_classification.train import (
            TrainConfig as DgTrain,
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
            DgData(num_samples=256, num_points=64, batch_size=32, val_fraction=0.2, seed=0, num_workers=0),
        )

        dg_paths = build_run_paths(
            track="pointcloud", lesson="lesson_02_dgcnn_toy_classification", run_name="smoke"
        )
        assert (dg_paths.run_dir / "config.json").is_file()
        assert (dg_paths.run_dir / "metrics.jsonl").is_file()
        assert (dg_paths.checkpoints_dir / "checkpoint.pt").is_file()

        # 4.15) PointCloud lesson: toy PointNet2 classification (torch-only).
        from tracks.pointcloud.lesson_03_pointnet2_toy_classification.data import DataConfig as P2Data
        from tracks.pointcloud.lesson_03_pointnet2_toy_classification.train import (
            TrainConfig as P2Train,
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
            P2Data(num_samples=256, num_points=64, batch_size=32, val_fraction=0.2, seed=0, num_workers=0),
        )

        p2_paths = build_run_paths(
            track="pointcloud", lesson="lesson_03_pointnet2_toy_classification", run_name="smoke"
        )
        assert (p2_paths.run_dir / "config.json").is_file()
        assert (p2_paths.run_dir / "metrics.jsonl").is_file()
        assert (p2_paths.checkpoints_dir / "checkpoint.pt").is_file()

    print("smoke_check: OK")
    print(f"- LogisticRegression train acc: {acc:.3f}")
    print(f"- WarmupCosine last lr: {opt.learning_rate:.6f}")
    print(f"- mean_squared_error(b, 0): {mse:.6f}")
    if ran_pytorch:
        print("- tracks: PyTorch lessons executed")
    else:
        print("- tracks: PyTorch lessons skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
