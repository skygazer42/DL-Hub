import pytest

torch = pytest.importorskip("torch")


def test_vision_monocular_depth_shapes_metadata_and_loss_smoke() -> None:
    from tracks.vision.lesson_19_synthetic_monocular_depth_estimation.data import (
        DataConfig,
        SyntheticMonocularDepthDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_19_synthetic_monocular_depth_estimation.model import (
        DepthRegressor,
        ModelConfig,
    )

    cfg = DataConfig(
        num_samples=48,
        batch_size=4,
        image_size=32,
        val_fraction=0.25,
        seed=7,
        num_workers=0,
        near_depth=0.2,
        far_depth=0.9,
        min_layers=2,
        max_layers=4,
        add_gradient_background=True,
        noise_std=0.01,
    )
    ds = SyntheticMonocularDepthDataset(cfg)
    image, target = ds[0]

    assert tuple(image.shape) == (1, 32, 32)
    assert set(target.keys()) == {"depth", "occlusion", "layer_ids"}
    assert tuple(target["depth"].shape) == (1, 32, 32)
    assert tuple(target["occlusion"].shape) == (1, 32, 32)
    assert tuple(target["layer_ids"].shape) == (1, 32, 32)
    assert image.dtype == torch.float32
    assert target["depth"].dtype == torch.float32
    assert target["occlusion"].dtype == torch.float32
    assert target["layer_ids"].dtype == torch.long
    assert float(target["depth"].min().item()) >= 0.2
    assert float(target["depth"].max().item()) <= 0.9
    assert target["occlusion"].max().item() <= 1.0
    assert target["occlusion"].min().item() >= 0.0

    train_loader, _ = get_dataloaders(cfg)
    batch_images, batch_targets = next(iter(train_loader))
    assert tuple(batch_images.shape) == (4, 1, 32, 32)
    assert tuple(batch_targets["depth"].shape) == (4, 1, 32, 32)

    model = DepthRegressor(ModelConfig(in_channels=1, hidden_channels=16, num_blocks=3))
    pred = model(batch_images)
    assert tuple(pred.shape) == (4, 1, 32, 32)

    loss = torch.nn.functional.l1_loss(pred, batch_targets["depth"])
    assert torch.isfinite(loss)
    loss.backward()


def test_vision_monocular_depth_dataset_is_deterministic_for_same_seed() -> None:
    from tracks.vision.lesson_19_synthetic_monocular_depth_estimation.data import (
        DataConfig,
        SyntheticMonocularDepthDataset,
    )

    cfg = DataConfig(num_samples=8, image_size=24, seed=11, noise_std=0.0)
    ds_a = SyntheticMonocularDepthDataset(cfg)
    ds_b = SyntheticMonocularDepthDataset(cfg)

    image_a, target_a = ds_a[3]
    image_b, target_b = ds_b[3]

    assert torch.allclose(image_a, image_b)
    assert torch.allclose(target_a["depth"], target_b["depth"])
    assert torch.equal(target_a["layer_ids"], target_b["layer_ids"])
