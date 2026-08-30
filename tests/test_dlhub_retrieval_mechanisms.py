import importlib

import torch


def _build(package: str, family: str):
    module = importlib.import_module(f"dlhub.vision.{package}.{family}")
    builders = [
        value
        for name, value in vars(module).items()
        if name.startswith("build_")
        and callable(value)
        and getattr(value, "__module__", None) == module.__name__
    ]
    assert len(builders) == 1
    return builders[0](
        in_channels=3,
        variant=f"{family}_tiny",
        width_mult=0.5,
    )


def test_context_conditioned_retrieval_families_use_supplied_context() -> None:
    image = torch.randn(2, 3, 24, 24)
    families = (
        ("image_retrieval", "clipret"),
        ("visual_place_recognition", "geoclip_vpr"),
        ("fine_grained_retrieval", "fgclip_retr"),
        ("fine_grained_retrieval", "prompt_fgret"),
    )

    for package, family in families:
        model = _build(package, family).eval()
        with torch.no_grad():
            first = model(image, context=torch.zeros(2, 64))["embedding"]
            second = model(image, context=torch.ones(2, 64))["embedding"]
        assert model.uses_context
        assert not torch.allclose(first, second)


def test_vlad_assignments_and_attention_weights_are_normalized() -> None:
    image = torch.randn(2, 3, 24, 24)
    vlad = _build("image_retrieval", "netvlad")
    attention = _build("image_retrieval", "arc")

    vlad(image)
    attention(image)

    assert vlad.pool.last_assignment is not None
    torch.testing.assert_close(
        vlad.pool.last_assignment.sum(dim=1),
        torch.ones_like(vlad.pool.last_assignment[:, 0]),
    )
    assert attention.pool.last_attention is not None
    torch.testing.assert_close(
        attention.pool.last_attention.sum(dim=-1),
        torch.ones(2, 1),
    )


def test_pairwise_scoring_differs_from_plain_cosine_and_backpropagates() -> None:
    image = torch.randn(2, 3, 24, 24)
    gallery = torch.randn(3, 3, 24, 24)
    pairwise = _build("image_retrieval", "pairret")

    output = pairwise(image, gallery)
    cosine = output["embedding"] @ output["gallery_embedding"].t()

    assert not torch.allclose(output["similarity"], cosine)
    output["similarity"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in pairwise.parameters())


def test_transformer_and_selective_scan_retrieval_paths_execute() -> None:
    image = torch.randn(2, 3, 24, 24)
    transformer = _build("image_retrieval", "transformerret")
    selective_scan = _build("visual_place_recognition", "mambavpr")

    transformer_embedding = transformer(image)["embedding"]
    scan_embedding = selective_scan(image)["embedding"]

    assert transformer.mechanism == "spatial-token-transformer"
    assert selective_scan.mechanism == "spatial-selective-scan"
    assert transformer_embedding.shape == scan_embedding.shape == (2, 64)
    assert not torch.allclose(transformer_embedding, scan_embedding)
