import pytest


torch = pytest.importorskip("torch")


def test_pointcloud_lesson_01_dataloaders_and_forward_smoke() -> None:
    from tracks.pointcloud.lesson_01_pointnet_toy_classification.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_01_pointnet_toy_classification.model import ModelConfig, PointNetClassifier

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=32, num_points=64, batch_size=8, val_fraction=0.2, seed=0, num_workers=0)
    )
    points, labels = next(iter(train_loader))
    assert points.shape == (8, 64, 3)
    assert labels.shape == (8,)

    model = PointNetClassifier(ModelConfig(hidden_features=32, num_classes=2, dropout=0.0))
    logits = model(points)
    assert logits.shape == (8, 2)


def test_pointcloud_lesson_02_dgcnn_forward_smoke() -> None:
    from tracks.pointcloud.lesson_02_dgcnn_toy_classification.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_02_dgcnn_toy_classification.model import DGCNNClassifier, ModelConfig

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=32, num_points=32, batch_size=4, val_fraction=0.2, seed=0, num_workers=0)
    )
    points, labels = next(iter(train_loader))
    assert points.shape == (4, 32, 3)
    assert labels.shape == (4,)

    model = DGCNNClassifier(ModelConfig(k=5, hidden_features=16, dropout=0.0, num_classes=2, dynamic_graph=True))
    logits = model(points)
    assert logits.shape == (4, 2)


def test_pointcloud_lesson_03_pointnet2_forward_smoke() -> None:
    from tracks.pointcloud.lesson_03_pointnet2_toy_classification.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_03_pointnet2_toy_classification.model import ModelConfig, PointNet2Classifier

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=32, num_points=64, batch_size=4, val_fraction=0.2, seed=0, num_workers=0)
    )
    points, labels = next(iter(train_loader))
    assert points.shape == (4, 64, 3)
    assert labels.shape == (4,)

    model = PointNet2Classifier(
        ModelConfig(
            npoint1=16,
            k1=8,
            npoint2=4,
            k2=4,
            hidden_features=32,
            dropout=0.0,
            num_classes=2,
        )
    )
    logits = model(points)
    assert logits.shape == (4, 2)


def test_pointcloud_lesson_05_pointnet_partseg_forward_smoke() -> None:
    from tracks.pointcloud.lesson_05_pointnet_toy_partseg.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_05_pointnet_toy_partseg.model import ModelConfig, PointNetPartSeg

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=16, num_points=64, batch_size=4, val_fraction=0.2, seed=0, num_workers=0)
    )
    points, labels = next(iter(train_loader))
    assert points.shape == (4, 64, 3)
    assert labels.shape == (4, 64)

    model = PointNetPartSeg(ModelConfig(in_channels=3, hidden_features=32, num_classes=2, dropout=0.0))
    logits = model(points)
    assert logits.shape == (4, 64, 2)


def test_pointcloud_lesson_06_dgcnn_partseg_forward_smoke() -> None:
    from tracks.pointcloud.lesson_06_dgcnn_toy_partseg.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_06_dgcnn_toy_partseg.model import DGCNNPartSeg, ModelConfig

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=16, num_points=48, batch_size=2, val_fraction=0.2, seed=0, num_workers=0)
    )
    points, labels = next(iter(train_loader))
    assert points.shape == (2, 48, 3)
    assert labels.shape == (2, 48)

    model = DGCNNPartSeg(ModelConfig(k=5, hidden_features=16, dropout=0.0, num_classes=2, dynamic_graph=True))
    logits = model(points)
    assert logits.shape == (2, 48, 2)


def test_pointcloud_lesson_07_pointnet_reconstruction_forward_loss_backward_smoke() -> None:
    from dlhub.pointcloud.ops import chamfer_distance
    from tracks.pointcloud.lesson_07_pointnet_toy_reconstruction.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_07_pointnet_toy_reconstruction.model import ModelConfig, build_model

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=32,
            num_points=48,
            batch_size=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.03,
            p_sphere=0.5,
        )
    )
    noisy, clean = next(iter(train_loader))
    assert noisy.shape == (4, 48, 3)
    assert clean.shape == (4, 48, 3)

    model = build_model(
        ModelConfig(
            in_channels=3,
            num_points=48,
            arch="pointnet_ae:pointnet_ae_tiny",
            variant="",
            dropout=0.0,
        )
    )
    pred = model(noisy)
    assert pred.shape == (4, 48, 3)
    loss = chamfer_distance(pred, clean)
    assert torch.isfinite(loss)
    loss.backward()


def test_pointcloud_lesson_08_partseg_zoo_build_forward_smoke() -> None:
    from tracks.pointcloud.lesson_08_pointcloud_partseg_zoo_toy.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_08_pointcloud_partseg_zoo_toy.model import ModelConfig, build_model

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=16, num_points=64, batch_size=2, val_fraction=0.2, seed=0, num_workers=0)
    )
    points, labels = next(iter(train_loader))
    assert points.shape == (2, 64, 3)
    assert labels.shape == (2, 64)

    for arch in ["pointnet", "dgcnn_static"]:
        model = build_model(
            ModelConfig(
                arch=arch,
                in_channels=3,
                num_classes=2,
                num_points=64,
                hidden_features=16,
                dropout=0.0,
                k=5,
                dynamic_graph=True,
            )
        )
        logits = model(points)
        assert logits.shape == (2, 64, 2)


def test_pointcloud_lesson_09_simclr_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_09_pointcloud_selfsupervised_simclr.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_09_pointcloud_selfsupervised_simclr.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.simclr import nt_xent_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=64,
            batch_size=8,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.1,
        )
    )
    v1, v2, y = next(iter(train_loader))
    assert v1.shape == (8, 64, 3)
    assert v2.shape == (8, 64, 3)
    assert y.shape == (8,)

    model = build_model(
        ModelConfig(
            arch="simclr_pointnet:simclr_pointnet_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
        )
    )
    out1 = model(v1)
    out2 = model(v2)
    assert set(out1.keys()) == {"h", "z"}
    assert out1["z"].ndim == 2 and out2["z"].ndim == 2
    loss = nt_xent_loss(out1["z"], out2["z"], temperature=0.2)
    assert torch.isfinite(loss)
    loss.backward()


def test_pointcloud_lesson_10_pointmae_ssl_forward_loss_backward_smoke() -> None:
    from dlhub.pointcloud.ops import chamfer_distance
    from tracks.pointcloud.lesson_10_pointcloud_selfsupervised_pointmae.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_10_pointcloud_selfsupervised_pointmae.model import ModelConfig, build_model

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=96,
            batch_size=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.0,
        )
    )
    points, y = next(iter(train_loader))
    assert points.shape == (4, 96, 3)
    assert y.shape == (4,)

    model = build_model(
        ModelConfig(
            arch="pointmae:pointmae_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
        )
    )
    out = model(points, mask_ratio=0.6)
    pred = out["pred"]
    target = out["target"]
    assert pred.ndim == 4 and target.ndim == 4
    assert pred.shape == target.shape
    loss = chamfer_distance(pred.reshape(-1, pred.shape[-2], 3), target.reshape(-1, target.shape[-2], 3))
    assert torch.isfinite(loss)
    loss.backward()


def test_pointcloud_lesson_11_byol_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_11_pointcloud_selfsupervised_byol.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_11_pointcloud_selfsupervised_byol.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.byol import byol_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=64,
            batch_size=8,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.1,
        )
    )
    v1, v2, y = next(iter(train_loader))
    assert v1.shape == (8, 64, 3)
    assert v2.shape == (8, 64, 3)
    assert y.shape == (8,)

    model = build_model(
        ModelConfig(
            arch="byol_pointnet:byol_pointnet_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
        )
    )
    out1 = model.forward_online(v1)
    out2 = model.forward_online(v2)
    tgt1 = model.forward_target(v1)
    tgt2 = model.forward_target(v2)
    assert set(out1.keys()) == {"h", "z", "p"}
    assert set(tgt1.keys()) == {"h", "z"}

    loss = byol_loss(out1["p"], tgt2["z"], out2["p"], tgt1["z"])
    assert torch.isfinite(loss)
    loss.backward()
    model.update_target(ema_decay=0.99)


def test_pointcloud_lesson_12_vicreg_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_12_pointcloud_selfsupervised_vicreg.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_12_pointcloud_selfsupervised_vicreg.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.vicreg import vicreg_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=64,
            batch_size=8,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.1,
        )
    )
    v1, v2, y = next(iter(train_loader))
    assert v1.shape == (8, 64, 3)
    assert v2.shape == (8, 64, 3)
    assert y.shape == (8,)

    model = build_model(
        ModelConfig(
            arch="vicreg_pointnet:vicreg_pointnet_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
        )
    )
    out1 = model(v1)
    out2 = model(v2)
    assert set(out1.keys()) == {"h", "z"}
    loss = vicreg_loss(out1["z"], out2["z"])
    assert torch.isfinite(loss)
    loss.backward()


def test_pointcloud_lesson_13_ssl_linear_probe_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_13_pointcloud_ssl_linear_probe.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_13_pointcloud_ssl_linear_probe.model import ModelConfig, build_model

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=64, num_points=64, batch_size=8, val_fraction=0.2, seed=0, num_workers=0)
    )
    points, labels = next(iter(train_loader))
    assert points.shape == (8, 64, 3)
    assert labels.shape == (8,)

    model = build_model(
        ModelConfig(
            ssl_arch="simclr_pointnet:simclr_pointnet_tiny",
            ssl_dropout=0.0,
            in_channels=3,
            num_classes=2,
            freeze_ssl=True,
        )
    )
    logits = model(points)
    assert logits.shape == (8, 2)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()


def test_pointcloud_lesson_14_moco_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_14_pointcloud_selfsupervised_moco.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_14_pointcloud_selfsupervised_moco.model import ModelConfig, build_model

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=64,
            batch_size=8,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.1,
        )
    )
    v1, v2, y = next(iter(train_loader))
    assert v1.shape == (8, 64, 3)
    assert v2.shape == (8, 64, 3)
    assert y.shape == (8,)

    model = build_model(
        ModelConfig(
            arch="moco_pointnet:moco_pointnet_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
            queue_size=64,
        )
    )
    model.momentum_update_key_encoder(ema_decay=0.99)
    out = model(v1, v2, temperature=0.2)
    assert set(out.keys()) == {"q", "k", "logits", "labels"}
    loss = torch.nn.functional.cross_entropy(out["logits"], out["labels"])
    assert torch.isfinite(loss)
    loss.backward()
    model.dequeue_and_enqueue(out["k"])


def test_pointcloud_lesson_15_simsiam_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_15_pointcloud_selfsupervised_simsiam.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_15_pointcloud_selfsupervised_simsiam.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.simsiam import simsiam_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=64,
            batch_size=8,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.1,
        )
    )
    v1, v2, y = next(iter(train_loader))
    assert v1.shape == (8, 64, 3)
    assert v2.shape == (8, 64, 3)
    assert y.shape == (8,)

    model = build_model(
        ModelConfig(
            arch="simsiam_pointnet:simsiam_pointnet_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
        )
    )
    o1 = model(v1)
    o2 = model(v2)
    assert set(o1.keys()) == {"h", "z", "p"}
    loss = simsiam_loss(o1["p"], o2["z"], o2["p"], o1["z"])
    assert torch.isfinite(loss)
    loss.backward()


def test_pointcloud_lesson_16_swav_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_16_pointcloud_selfsupervised_swav.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_16_pointcloud_selfsupervised_swav.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.swav import swav_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=64,
            batch_size=8,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.1,
        )
    )
    v1, v2, y = next(iter(train_loader))
    assert v1.shape == (8, 64, 3)
    assert v2.shape == (8, 64, 3)
    assert y.shape == (8,)

    model = build_model(
        ModelConfig(
            arch="swav_pointnet:swav_pointnet_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
            num_prototypes=32,
        )
    )
    o1 = model(v1)
    o2 = model(v2)
    assert set(o1.keys()) == {"h", "z", "scores"}
    loss = swav_loss(o1["scores"], o2["scores"], temperature=0.1, sinkhorn_epsilon=0.05, sinkhorn_iters=2)
    assert torch.isfinite(loss)
    loss.backward()
    model.normalize_prototypes()


def test_pointcloud_lesson_17_barlowtwins_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_17_pointcloud_selfsupervised_barlowtwins.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_17_pointcloud_selfsupervised_barlowtwins.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.barlowtwins import barlow_twins_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=64,
            batch_size=8,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.1,
        )
    )
    v1, v2, y = next(iter(train_loader))
    assert v1.shape == (8, 64, 3)
    assert v2.shape == (8, 64, 3)
    assert y.shape == (8,)

    model = build_model(
        ModelConfig(
            arch="barlowtwins_pointnet:barlowtwins_pointnet_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
        )
    )
    o1 = model(v1)
    o2 = model(v2)
    assert set(o1.keys()) == {"h", "z"}
    loss = barlow_twins_loss(o1["z"], o2["z"], lambda_offdiag=0.005)
    assert torch.isfinite(loss)
    loss.backward()


def test_pointcloud_lesson_18_dino_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_18_pointcloud_selfsupervised_dino.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_18_pointcloud_selfsupervised_dino.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.dino import dino_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=64,
            batch_size=8,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.1,
        )
    )
    v1, v2, y = next(iter(train_loader))
    assert v1.shape == (8, 64, 3)
    assert v2.shape == (8, 64, 3)
    assert y.shape == (8,)

    model = build_model(
        ModelConfig(
            arch="dino_pointnet:dino_pointnet_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
            out_dim=64,
        )
    )

    with torch.no_grad():
        t1 = model.forward_teacher(v1)["logits"]
        t2 = model.forward_teacher(v2)["logits"]
    s1 = model.forward_student(v1)["logits"]
    s2 = model.forward_student(v2)["logits"]
    loss = dino_loss(
        [s1, s2],
        [t1, t2],
        student_temperature=0.1,
        teacher_temperature=0.04,
        center=model.center,
    )
    assert torch.isfinite(loss)
    loss.backward()
    model.update_center([t1, t2], center_momentum=0.9)
    model.momentum_update_teacher(ema_decay=0.99)


def test_pointcloud_lesson_19_dinov2_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_19_pointcloud_selfsupervised_dinov2.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_19_pointcloud_selfsupervised_dinov2.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.dinov2 import dino_cross_view_loss, ibot_patch_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=96,
            batch_size=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.1,
        )
    )
    v1, v2, y = next(iter(train_loader))
    assert v1.shape == (4, 96, 3)
    assert v2.shape == (4, 96, 3)
    assert y.shape == (4,)

    model = build_model(
        ModelConfig(
            arch="dinov2_pointmae:dinov2_pointmae_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
            out_dim=64,
        )
    )
    with torch.no_grad():
        t1 = model.forward_teacher(v1)
        t2 = model.forward_teacher(v2)

    s1 = model.forward_student(v1, mask_ratio=0.5)
    s2 = model.forward_student(v2, mask_ratio=0.5)

    loss_cls = dino_cross_view_loss(
        [s1["cls_logits"], s2["cls_logits"]],
        [t1["cls_logits"], t2["cls_logits"]],
        student_temperature=0.1,
        teacher_temperature=0.04,
        center=model.center_cls,
    )
    loss_patch = 0.5 * (
        ibot_patch_loss(
            s1["patch_logits"],
            t1["patch_logits"],
            s1["mask_idx"],
            student_temperature=0.1,
            teacher_temperature=0.04,
            center=model.center_patch,
        )
        + ibot_patch_loss(
            s2["patch_logits"],
            t2["patch_logits"],
            s2["mask_idx"],
            student_temperature=0.1,
            teacher_temperature=0.04,
            center=model.center_patch,
        )
    )
    loss = loss_cls + loss_patch
    assert torch.isfinite(loss)
    loss.backward()

    model.update_centers(
        teacher_cls_logits=[t1["cls_logits"], t2["cls_logits"]],
        teacher_patch_logits=[t1["patch_logits"], t2["patch_logits"]],
        center_momentum=0.9,
    )
    model.momentum_update_teacher(ema_decay=0.99)


def test_pointcloud_lesson_20_ijepa_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_20_pointcloud_selfsupervised_ijepa.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_20_pointcloud_selfsupervised_ijepa.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.ijepa import ijepa_patch_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=96,
            batch_size=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.0,
        )
    )
    points, y = next(iter(train_loader))
    assert points.shape == (4, 96, 3)
    assert y.shape == (4,)

    model = build_model(
        ModelConfig(
            arch="ijepa_pointmae:ijepa_pointmae_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
        )
    )
    with torch.no_grad():
        target_patch = model.forward_teacher(points)["patch"]
    out = model.forward_student(points, mask_ratio=0.5)
    loss = ijepa_patch_loss(out["pred"], target_patch, out["mask_idx"])
    assert torch.isfinite(loss)
    loss.backward()
    model.momentum_update_teacher(ema_decay=0.99)


def test_pointcloud_lesson_21_msn_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_21_pointcloud_selfsupervised_msn.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_21_pointcloud_selfsupervised_msn.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.msn import msn_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=96,
            batch_size=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.1,
        )
    )
    v1, v2, y = next(iter(train_loader))
    assert v1.shape == (4, 96, 3)
    assert v2.shape == (4, 96, 3)
    assert y.shape == (4,)

    model = build_model(
        ModelConfig(
            arch="msn_pointmae:msn_pointmae_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
            out_dim=64,
        )
    )
    with torch.no_grad():
        t1 = model.forward_teacher(v1)["cls_logits"]
        t2 = model.forward_teacher(v2)["cls_logits"]

    s1 = model.forward_student(v1, mask_ratio=0.5)["cls_logits"]
    s2 = model.forward_student(v2, mask_ratio=0.5)["cls_logits"]

    loss = msn_loss(
        student_logits=[s1, s2],
        teacher_logits=[t1, t2],
        student_temperature=0.1,
        teacher_temperature=0.04,
        center=model.center,
        entropy_weight=1.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    model.update_center([t1, t2], center_momentum=0.9)
    model.momentum_update_teacher(ema_decay=0.99)


def test_pointcloud_lesson_22_data2vec_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_22_pointcloud_selfsupervised_data2vec.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_22_pointcloud_selfsupervised_data2vec.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.data2vec import data2vec_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=96,
            batch_size=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            jitter_std=0.01,
            drop_p=0.1,
        )
    )
    v1, v2, y = next(iter(train_loader))
    assert v1.shape == (4, 96, 3)
    assert v2.shape == (4, 96, 3)
    assert y.shape == (4,)

    model = build_model(
        ModelConfig(
            arch="data2vec_pointmae:data2vec_pointmae_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
        )
    )
    with torch.no_grad():
        t1 = model.forward_teacher(v1)
        t2 = model.forward_teacher(v2)

    s1 = model.forward_student(v1, mask_ratio=0.5)
    s2 = model.forward_student(v2, mask_ratio=0.5)

    loss = 0.5 * (
        data2vec_loss(
            pred_cls=s1["pred_cls"],
            target_cls=t1["cls"],
            pred_patch=s1["pred_patch"],
            target_patch=t1["patch"],
            mask_idx=s1["mask_idx"],
            cls_weight=1.0,
            patch_weight=1.0,
        )
        + data2vec_loss(
            pred_cls=s2["pred_cls"],
            target_cls=t2["cls"],
            pred_patch=s2["pred_patch"],
            target_patch=t2["patch"],
            mask_idx=s2["mask_idx"],
            cls_weight=1.0,
            patch_weight=1.0,
        )
    )
    assert torch.isfinite(loss)
    loss.backward()
    model.momentum_update_teacher(ema_decay=0.99)


def test_pointcloud_lesson_23_ressl_ssl_forward_loss_backward_smoke() -> None:
    from tracks.pointcloud.lesson_23_pointcloud_selfsupervised_ressl.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_23_pointcloud_selfsupervised_ressl.model import ModelConfig, build_model
    from dlhub.pointcloud.selfsupervised.ressl import ressl_loss

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            num_points=64,
            batch_size=8,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
            strong_jitter_std=0.02,
            strong_drop_p=0.2,
            weak_jitter_std=0.005,
            weak_drop_p=0.0,
        )
    )
    v_strong, v_weak, y = next(iter(train_loader))
    assert v_strong.shape == (8, 64, 3)
    assert v_weak.shape == (8, 64, 3)
    assert y.shape == (8,)

    model = build_model(
        ModelConfig(
            arch="ressl_pointnet:ressl_pointnet_tiny",
            variant="",
            in_channels=3,
            dropout=0.0,
            queue_size=128,
        )
    )
    out = model(v_strong, v_weak, student_temperature=0.2, teacher_temperature=0.04)
    loss = ressl_loss(out["student_logits"], out["teacher_logits"])
    assert torch.isfinite(loss)
    loss.backward()
    model.momentum_update_teacher(ema_decay=0.99)
    model.dequeue_and_enqueue(out["teacher_z"])
