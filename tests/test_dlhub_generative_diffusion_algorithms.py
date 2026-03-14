import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported diffusion output type: {type(x)!r}")


@pytest.mark.parametrize(
    "builder_name,kwargs",
    [
        (
            "build_ddpm_diffusion",
            {
                "in_channels": 3,
                "image_size": 32,
                "latent_dim": 64,
                "num_classes": 10,
                "variant": "ddpm_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_score_sde_diffusion",
            {
                "in_channels": 3,
                "image_size": 32,
                "latent_dim": 64,
                "num_classes": 10,
                "variant": "score_sde_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_latent_diffusion_diffusion",
            {
                "in_channels": 3,
                "image_size": 32,
                "latent_dim": 64,
                "num_classes": 10,
                "variant": "latent_diffusion_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
        (
            "build_flow_matching_diffusion",
            {
                "in_channels": 3,
                "image_size": 32,
                "latent_dim": 64,
                "num_classes": 10,
                "variant": "flow_matching_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
        ),
    ],
)
def test_diffusion_algorithms_forward_smoke(builder_name: str, kwargs: dict) -> None:
    import dlhub.generative.diffusion as diffusion

    build = getattr(diffusion, builder_name)
    model = build(**kwargs)
    out = model.forward(batch_size=2)
    assert isinstance(out, dict)
    assert "sample" in out
    assert "pred_noise" in out
    assert tuple(out["sample"].shape) == (2, 3, 32, 32)
    assert tuple(out["pred_noise"].shape) == (2, 3, 32, 32)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
