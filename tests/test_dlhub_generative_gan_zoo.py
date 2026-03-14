import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in GAN zoo smoke: {type(x)!r}")


def test_gan_zoo_lists_24_families_3_variants() -> None:
    from dlhub.generative.gan_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 72
    assert "gan:dcgan_tiny" in arches
    assert "gan:lsgan_small" in arches
    assert "gan:wgangp_base" in arches
    assert "gan:cgan_tiny" in arches
    assert "gan:pix2pix_small" in arches
    assert "gan:stylegan2_base" in arches
    assert "gan:hingegan_tiny" in arches
    assert "gan:projection_gan_base" in arches
    assert "gan:cutgan_small" in arches
    assert "gan:stylegan3_base" in arches


@pytest.mark.parametrize(
    "arch_id",
    [
        "gan:dcgan_tiny",
        "gan:cgan_tiny",
        "gan:pix2pix_tiny",
        "gan:stylegan2_tiny",
    ],
)
def test_gan_zoo_build_and_forward_smoke(arch_id: str) -> None:
    from dlhub.generative.gan_zoo import build_local_model

    model = build_local_model(
        arch_id,
        in_channels=3,
        image_size=32,
        latent_dim=64,
        num_classes=10,
        width_mult=0.5,
        dropout=0.0,
    )
    out = model.forward(batch_size=2)
    assert isinstance(out, dict)
    assert "fake_images" in out and "fake_logits" in out
    assert tuple(out["fake_images"].shape) == (2, 3, 32, 32)
    assert tuple(out["fake_logits"].shape) == (2,)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
