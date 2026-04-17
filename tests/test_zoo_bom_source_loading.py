import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "expected_arch"),
    [
        ("dlhub.generative.diffusion_zoo", "diff:flux_tiny"),
        ("dlhub.multimodal.vlm_zoo", "vlm:agent_vl_tiny"),
        ("dlhub.vision.action_recognition_zoo", "dlacts:aagcn_tiny"),
    ],
)
def test_zoo_registries_include_bom_prefixed_family_modules(
    module_name: str, expected_arch: str
) -> None:
    module = importlib.import_module(module_name)
    arches = module.list_local_arches()
    assert expected_arch in arches
