import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported GAN output type: {type(x)!r}")


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        (
            "build_dcgan_gan",
            {
                "in_channels": 3,
                "image_size": 32,
                "latent_dim": 64,
                "num_classes": 10,
                "variant": "dcgan_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_wgangp_gan",
            {
                "in_channels": 3,
                "image_size": 32,
                "latent_dim": 64,
                "num_classes": 10,
                "variant": "wgangp_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_cgan_gan",
            {
                "in_channels": 3,
                "image_size": 32,
                "latent_dim": 64,
                "num_classes": 10,
                "variant": "cgan_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_stylegan2_gan",
            {
                "in_channels": 3,
                "image_size": 32,
                "latent_dim": 64,
                "num_classes": 10,
                "variant": "stylegan2_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_stylegan3_gan",
            {
                "in_channels": 3,
                "image_size": 32,
                "latent_dim": 64,
                "num_classes": 10,
                "variant": "stylegan3_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
    ],
)
def test_gan_algorithms_forward_smoke(builder_name: str, kwargs: dict) -> None:
    import dlhub.generative.gan as gan

    build = getattr(gan, builder_name)
    model = build(**kwargs)
    out = model.forward(batch_size=2)
    assert isinstance(out, dict)
    assert "fake_images" in out
    assert "fake_logits" in out
    assert tuple(out["fake_images"].shape) == (2, 3, 32, 32)
    assert tuple(out["fake_logits"].shape) == (2,)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
