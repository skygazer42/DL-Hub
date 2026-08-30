import importlib

import torch


REGISTRATION_FAMILIES = (
    ("pointnetlk", "build_pointnetlk_registrar"),
    ("dcp", "build_dcp_registrar"),
    ("regtr", "build_regtr_registrar"),
    ("rpmnet", "build_rpmnet_registrar"),
    ("deepgmr", "build_deepgmr_registrar"),
    ("spinreg", "build_spinreg_registrar"),
    ("cofinet_reg", "build_cofinet_reg_registrar"),
    ("geoformer_reg", "build_geoformer_reg_registrar"),
    ("predator_reg", "build_predator_reg_registrar"),
    ("mambareg", "build_mambareg_registrar"),
)


def _build(family: str, builder_name: str):
    module = importlib.import_module(f"dlhub.pointcloud.registration.{family}")
    builder = getattr(module, builder_name)
    return builder(variant=f"{family}_tiny", width_mult=0.5)


def test_registration_families_have_distinct_translation_sensitive_mechanisms() -> None:
    source = torch.randn(2, 24, 3)
    target = source.clone()
    shifted_target = target + torch.tensor([0.5, 0.0, 0.0])
    mechanisms = set()
    model_types = set()

    for family, builder_name in REGISTRATION_FAMILIES:
        torch.manual_seed(23)
        model = _build(family, builder_name).eval()
        with torch.no_grad():
            aligned_pose = model(source, target)["pose6d"]
            shifted_pose = model(source, shifted_target)["pose6d"]

        assert tuple(aligned_pose.shape) == (2, 6)
        assert torch.isfinite(aligned_pose).all()
        assert torch.all(shifted_pose[:, 0] - aligned_pose[:, 0] > 0.35)
        mechanisms.add(model.mechanism)
        model_types.add(type(model))

    assert len(mechanisms) == len(REGISTRATION_FAMILIES)
    assert len(model_types) == len(REGISTRATION_FAMILIES)


def test_registration_models_are_point_permutation_invariant_and_backpropagate() -> None:
    source = torch.randn(2, 20, 3)
    target = torch.randn(2, 20, 3)
    source_permutation = torch.randperm(source.shape[1])
    target_permutation = torch.randperm(target.shape[1])

    for family, builder_name in REGISTRATION_FAMILIES:
        torch.manual_seed(41)
        model = _build(family, builder_name)
        pose = model(source, target)["pose6d"]
        permuted_pose = model(
            source[:, source_permutation],
            target[:, target_permutation],
        )["pose6d"]
        torch.testing.assert_close(pose, permuted_pose, rtol=1e-4, atol=1e-5)

        pose.square().mean().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_registration_correspondence_state_is_normalized_and_observable() -> None:
    source = torch.randn(2, 18, 3)
    target = torch.randn(2, 22, 3)

    dcp = _build("dcp", "build_dcp_registrar")
    rpmnet = _build("rpmnet", "build_rpmnet_registrar")
    deepgmr = _build("deepgmr", "build_deepgmr_registrar")
    predator = _build("predator_reg", "build_predator_reg_registrar")

    dcp(source, target)
    rpmnet(source, target)
    deepgmr(source, target)
    predator(source, target)

    assert dcp.last_correspondence is not None
    torch.testing.assert_close(
        dcp.last_correspondence.sum(dim=-1),
        torch.ones(2, 18),
    )
    assert rpmnet.last_transport is not None
    torch.testing.assert_close(
        rpmnet.last_transport.sum(dim=-1),
        torch.ones(2, 18),
    )
    assert deepgmr.last_assignments is not None
    for assignments in deepgmr.last_assignments:
        torch.testing.assert_close(
            assignments.sum(dim=-1),
            torch.ones(assignments.shape[:2]),
        )
    assert predator.last_overlap is not None
    assert torch.all((predator.last_overlap >= 0.0) & (predator.last_overlap <= 1.0))
