import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in diffusion zoo contract: {type(x)!r}")


def test_diffusion_zoo_lists_12_families_3_variants() -> None:
    from dlhub.generative.diffusion_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 36
    assert "diff:ddpm_tiny" in arches
    assert "diff:ddim_small" in arches
    assert "diff:score_sde_base" in arches
    assert "diff:latent_diffusion_tiny" in arches
    assert "diff:edm_small" in arches
    assert "diff:flow_matching_base" in arches


@pytest.mark.parametrize(
    "arch_id",
    [
        "diff:ddpm_tiny",
        "diff:score_sde_tiny",
        "diff:latent_diffusion_tiny",
        "diff:flow_matching_tiny",
    ],
)
def test_diffusion_zoo_build_and_forward_contract(arch_id: str) -> None:
    from dlhub.generative.diffusion_zoo import build_local_model

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
    assert "sample" in out and "pred_noise" in out
    assert tuple(out["sample"].shape) == (2, 3, 32, 32)
    assert tuple(out["pred_noise"].shape) == (2, 3, 32, 32)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
