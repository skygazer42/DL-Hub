import torch

from dlhub.vision.layout_generation_zoo import build_local_model, list_local_arches
from dlhub.zoo_fidelity import FidelityLevel, fidelity_for_artifact


FAMILIES = (
    "layoutgan_baseline",
    "layoutvae_baseline",
    "layouttransformer",
    "bbox_generator",
    "poster_layout_net",
    "doc_layout_gen",
    "constraint_layout",
    "relation_layout",
    "diffusion_layout",
    "mamba_layout_gen",
)


def test_layout_generation_registry_exposes_three_configs_per_method() -> None:
    arches = list_local_arches()

    assert len(arches) == 30
    for family in FAMILIES:
        assert {f"layout:{family}_{size}" for size in ("tiny", "small", "base")} <= set(
            arches
        )
        assert (
            fidelity_for_artifact(f"dlhub/vision/layout_generation/{family}.py")
            is FidelityLevel.COMPACT
        )


def test_layout_generation_families_have_distinct_executable_mechanisms() -> None:
    x = torch.randn(1, 3, 16, 16)
    mechanisms = set()

    for family in FAMILIES:
        torch.manual_seed(17)
        model = build_local_model(
            f"layout:{family}_tiny",
            in_channels=3,
            width_mult=0.5,
        )
        model.eval()
        with torch.no_grad():
            output = model(x)
            shifted = model(x + 0.1)

        assert tuple(output.shape) == tuple(x.shape)
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, shifted)
        mechanisms.add(model.mechanism)

    assert len(mechanisms) == len(FAMILIES)


def test_layout_diffusion_time_and_vae_latent_are_observable() -> None:
    x = torch.randn(2, 3, 16, 16)
    diffusion = build_local_model(
        "layout:diffusion_layout_tiny", in_channels=3, width_mult=0.5
    ).eval()
    vae = build_local_model(
        "layout:layoutvae_baseline_tiny", in_channels=3, width_mult=0.5
    ).eval()

    with torch.no_grad():
        early = diffusion(x, timestep=0.1)
        late = diffusion(x, timestep=0.9)
        vae_output = vae(x)

    assert not torch.allclose(early, late)
    assert tuple(vae_output.shape) == tuple(x.shape)
    assert vae.last_kl.ndim == 0
    assert torch.isfinite(vae.last_kl)


def test_layout_transformer_and_selective_scan_backpropagate() -> None:
    x = torch.randn(2, 3, 16, 16)
    for arch in ("layout:layouttransformer_tiny", "layout:mamba_layout_gen_tiny"):
        model = build_local_model(arch, in_channels=3, width_mult=0.5)
        output = model(x)
        loss = output.square().mean()
        loss.backward()

        assert any(parameter.grad is not None for parameter in model.parameters())
