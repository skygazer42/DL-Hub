"""GNN track smoke checks (torch-only)."""


def run() -> None:
    from dlhub.paths import build_run_paths

    # 4.3) GNN lesson (torch-only, fully synthetic).
    from tracks.gnn.lesson_01_compact_graph_classification.data import DataConfig as GnnData
    from tracks.gnn.lesson_01_compact_graph_classification.train import TrainConfig as GnnTrain
    from tracks.gnn.lesson_01_compact_graph_classification.train import run_training as run_gnn

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
        GnnData(
            num_graphs=64, num_nodes=10, batch_size=16, val_fraction=0.2, seed=0, num_workers=0
        ),
    )

    gnn_paths = build_run_paths(
        track="gnn", lesson="lesson_01_compact_graph_classification", run_name="smoke"
    )
    assert (gnn_paths.run_dir / "config.json").is_file()
    assert (gnn_paths.run_dir / "metrics.jsonl").is_file()
    assert (gnn_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.4) GNN lesson: GIN (compact, torch-only).
    from tracks.gnn.lesson_02_gin_compact_graph_classification.data import DataConfig as GinData
    from tracks.gnn.lesson_02_gin_compact_graph_classification.train import TrainConfig as GinTrain
    from tracks.gnn.lesson_02_gin_compact_graph_classification.train import run_training as run_gin

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
        GinData(
            num_graphs=64, num_nodes=10, batch_size=16, val_fraction=0.2, seed=0, num_workers=0
        ),
    )

    gin_paths = build_run_paths(
        track="gnn", lesson="lesson_02_gin_compact_graph_classification", run_name="smoke"
    )
    assert (gin_paths.run_dir / "config.json").is_file()
    assert (gin_paths.run_dir / "metrics.jsonl").is_file()
    assert (gin_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.5) GNN lesson: GAT (compact, torch-only).
    from tracks.gnn.lesson_03_gat_compact_graph_classification.data import DataConfig as GatData
    from tracks.gnn.lesson_03_gat_compact_graph_classification.train import TrainConfig as GatTrain
    from tracks.gnn.lesson_03_gat_compact_graph_classification.train import run_training as run_gat

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
        GatData(
            num_graphs=64, num_nodes=10, batch_size=16, val_fraction=0.2, seed=0, num_workers=0
        ),
    )

    gat_paths = build_run_paths(
        track="gnn", lesson="lesson_03_gat_compact_graph_classification", run_name="smoke"
    )
    assert (gat_paths.run_dir / "config.json").is_file()
    assert (gat_paths.run_dir / "metrics.jsonl").is_file()
    assert (gat_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.6) GNN lesson: Cora node classification (GCN).
    from tracks.gnn.lesson_04_cora_node_classification_gcn.train import TrainConfig as CoraTrain
    from tracks.gnn.lesson_04_cora_node_classification_gcn.train import run_training as run_cora

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
    from tracks.gnn.lesson_05_label_propagation_cora.train import TrainConfig as LpTrain
    from tracks.gnn.lesson_05_label_propagation_cora.train import run_training as run_lp

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

    lp_paths = build_run_paths(
        track="gnn", lesson="lesson_05_label_propagation_cora", run_name="smoke"
    )
    assert (lp_paths.run_dir / "config.json").is_file()
    assert (lp_paths.run_dir / "metrics.jsonl").is_file()
    assert (lp_paths.run_dir / "preds.pt").is_file()
    assert (lp_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.8) GNN lesson: Cora node classification (GraphSAGE, full-batch).
    from tracks.gnn.lesson_06_graphsage_cora.train import TrainConfig as SageTrain
    from tracks.gnn.lesson_06_graphsage_cora.train import run_training as run_sage

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

    sage_paths = build_run_paths(
        track="gnn", lesson="lesson_06_graphsage_cora", run_name="smoke"
    )
    assert (sage_paths.run_dir / "config.json").is_file()
    assert (sage_paths.run_dir / "metrics.jsonl").is_file()
    assert (sage_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.9) GNN lesson: Karate SDNE embeddings (torch-only, tiny).
    from tracks.gnn.lesson_07_sdne_karate_embedding.train import TrainConfig as SdneTrain
    from tracks.gnn.lesson_07_sdne_karate_embedding.train import run_training as run_sdne

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

    sdne_paths = build_run_paths(
        track="gnn", lesson="lesson_07_sdne_karate_embedding", run_name="smoke"
    )
    assert (sdne_paths.run_dir / "config.json").is_file()
    assert (sdne_paths.run_dir / "metrics.jsonl").is_file()
    assert (sdne_paths.run_dir / "embeddings.pt").is_file()
    assert (sdne_paths.checkpoints_dir / "checkpoint.pt").is_file()

    # 4.10) GNN lesson: Karate LINE embeddings (torch-only, tiny).
    from tracks.gnn.lesson_08_line_karate_embedding.train import TrainConfig as LineTrain
    from tracks.gnn.lesson_08_line_karate_embedding.train import run_training as run_line

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

    line_paths = build_run_paths(
        track="gnn", lesson="lesson_08_line_karate_embedding", run_name="smoke"
    )
    assert (line_paths.run_dir / "config.json").is_file()
    assert (line_paths.run_dir / "metrics.jsonl").is_file()
    assert (line_paths.run_dir / "embeddings.pt").is_file()
    assert (line_paths.checkpoints_dir / "checkpoint.pt").is_file()
