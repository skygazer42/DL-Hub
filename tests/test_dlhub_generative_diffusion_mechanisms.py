import torch

from dlhub.generative.diffusion_zoo import build_local_model, list_local_arches
from dlhub.zoo_fidelity import FidelityLevel, fidelity_for_artifact


def _build(arch_id: str, *, num_classes: int = 0):
    return build_local_model(
        arch_id,
        in_channels=3,
        image_size=16,
        latent_dim=32,
        num_classes=num_classes,
        width_mult=0.25,
        dropout=0.0,
    )


def test_diffusion_architectures_are_deterministic_with_explicit_state_and_time_sensitive() -> None:
    x_t = torch.randn(2, 3, 16, 16)
    representatives = (
        "diff:ddpm_tiny",
        "diff:dit_tiny",
        "diff:latent_diffusion_tiny",
    )
    architectures = set()

    for arch_id in representatives:
        model = _build(arch_id).eval()
        with torch.no_grad():
            early = model(x_t=x_t, timesteps=0.1)
            repeated = model(x_t=x_t, timesteps=0.1)
            late = model(x_t=x_t, timesteps=0.9)
            changed_state = model(x_t=x_t + 0.2, timesteps=0.1)

        torch.testing.assert_close(early["sample"], repeated["sample"])
        assert not torch.allclose(early["pred_noise"], late["pred_noise"])
        assert not torch.allclose(early["sample"], late["sample"])
        assert not torch.allclose(early["pred_noise"], changed_state["pred_noise"])
        architectures.add(model.architecture)

    assert architectures == {
        "spatial-convolutional-denoiser",
        "patch-transformer-denoiser",
        "latent-autoencoder-denoiser",
    }


def test_conditional_diffusion_uses_labels_and_iterative_schedule() -> None:
    model = _build("diff:stable_diffusion_tiny", num_classes=4).eval()
    initial_noise = torch.randn(2, 3, 16, 16)

    with torch.no_grad():
        first = model(
            x_t=initial_noise,
            timesteps=0.5,
            labels=torch.zeros(2, dtype=torch.long),
        )
        second = model(
            x_t=initial_noise,
            timesteps=0.5,
            labels=torch.ones(2, dtype=torch.long),
        )
        one_step = model.sample(
            initial_noise=initial_noise,
            labels=torch.zeros(2, dtype=torch.long),
            num_steps=1,
        )
        four_steps = model.sample(
            initial_noise=initial_noise,
            labels=torch.zeros(2, dtype=torch.long),
            num_steps=4,
        )

    assert model.use_condition
    assert not torch.allclose(first["pred_noise"], second["pred_noise"])
    assert not torch.allclose(one_step, four_steps)


def test_ddim_step_scale_changes_the_update_and_denoiser_backpropagates() -> None:
    x_t = torch.randn(2, 3, 16, 16)
    timesteps = torch.tensor([0.25, 0.75])
    torch.manual_seed(31)
    ddpm = _build("diff:ddpm_tiny")
    torch.manual_seed(31)
    ddim = _build("diff:ddim_tiny")

    ddpm_output = ddpm(x_t=x_t, timesteps=timesteps)
    ddim_output = ddim(x_t=x_t, timesteps=timesteps)

    torch.testing.assert_close(ddpm_output["pred_noise"], ddim_output["pred_noise"])
    assert not torch.allclose(ddpm_output["sample"], ddim_output["sample"])
    ddpm_output["pred_noise"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in ddpm.parameters())


def test_every_diffusion_family_tiny_registration_accepts_explicit_noisy_inputs() -> None:
    tiny_arches = [arch for arch in list_local_arches() if arch.endswith("_tiny")]
    x_t = torch.randn(1, 3, 16, 16)

    assert len(tiny_arches) == 32
    for arch_id in tiny_arches:
        model = _build(arch_id, num_classes=3).eval()
        with torch.no_grad():
            output = model(
                x_t=x_t,
                timesteps=torch.tensor([0.4]),
                labels=torch.tensor([1]) if model.use_condition else None,
            )
        assert output["sample"].shape == x_t.shape
        assert output["pred_noise"].shape == x_t.shape
        assert torch.isfinite(output["sample"]).all()


def test_diffusion_fidelity_distinguishes_representatives_from_shared_labels() -> None:
    assert (
        fidelity_for_artifact("dlhub/generative/diffusion/ddpm.py")
        is FidelityLevel.COMPACT
    )
    assert (
        fidelity_for_artifact("dlhub/generative/diffusion/sdxl.py")
        is FidelityLevel.BASELINE_ALIAS
    )
