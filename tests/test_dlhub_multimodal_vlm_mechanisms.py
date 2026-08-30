import torch

from dlhub.multimodal.vlm_zoo import build_local_model, list_local_arches
from dlhub.zoo_fidelity import FidelityLevel, fidelity_for_artifact


def _build(arch_id: str):
    return build_local_model(
        arch_id,
        image_size=16,
        vocab_size=64,
        seq_len=6,
        embed_dim=64,
        num_classes=5,
        width_mult=0.5,
        dropout=0.0,
    )


def test_vlm_modes_are_deterministic_with_real_inputs_and_modality_sensitive() -> None:
    images = torch.randn(2, 3, 16, 16)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    representatives = (
        "vlm:clip_tiny",
        "vlm:vilt_tiny",
        "vlm:blip_tiny",
        "vlm:flamingo_tiny",
    )
    mechanisms = set()

    for arch_id in representatives:
        model = _build(arch_id).eval()
        with torch.no_grad():
            first = model(images=images, input_ids=input_ids)
            repeated = model(images=images, input_ids=input_ids)
            changed_image = model(images=images + 0.25, input_ids=input_ids)
            changed_text = model(images=images, input_ids=(input_ids + 7) % 64)

        torch.testing.assert_close(first["logits"], repeated["logits"])
        assert not torch.allclose(first["image_embed"], changed_image["image_embed"])
        assert not torch.allclose(first["text_embed"], changed_text["text_embed"])
        assert not torch.allclose(first["logits"], changed_image["logits"])
        assert not torch.allclose(first["logits"], changed_text["logits"])
        mechanisms.add(model.mechanism)

    assert mechanisms == {
        "contrastive-dual-encoder",
        "joint-multimodal-transformer",
        "text-to-image-cross-attention",
        "query-token-vision-language-bridge",
    }


def test_vlm_cross_attention_and_token_generation_are_observable() -> None:
    images = torch.randn(2, 3, 16, 16)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    fusion = _build("vlm:blip_tiny")
    bridge = _build("vlm:flamingo_tiny")

    fusion_output = fusion(images=images, input_ids=input_ids)
    bridge_output = bridge(images=images, input_ids=input_ids)

    assert fusion.last_cross_attention is not None
    torch.testing.assert_close(
        fusion.last_cross_attention.sum(dim=-1),
        torch.ones_like(fusion.last_cross_attention[..., 0]),
    )
    assert bridge.last_cross_attention is not None
    assert torch.isfinite(bridge.last_cross_attention).all()
    assert bridge_output["token_logits"].shape == (2, 6, 64)
    assert not torch.allclose(
        bridge_output["token_logits"][:, 0],
        bridge_output["token_logits"][:, 1],
    )

    loss = fusion_output["token_logits"].square().mean()
    loss = loss + bridge_output["token_logits"].square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in fusion.parameters())
    assert any(parameter.grad is not None for parameter in bridge.parameters())


def test_instruction_tokens_change_instruction_aware_vlm_outputs() -> None:
    model = _build("vlm:llava_tiny").eval()
    images = torch.randn(2, 3, 16, 16)
    input_ids = torch.randint(0, 64, (2, 6))

    with torch.no_grad():
        first = model(
            images=images,
            input_ids=input_ids,
            instruction_ids=torch.zeros(2, 4, dtype=torch.long),
        )
        second = model(
            images=images,
            input_ids=input_ids,
            instruction_ids=torch.ones(2, 4, dtype=torch.long),
        )

    assert model.use_instruction
    assert not torch.allclose(first["text_embed"], second["text_embed"])
    assert not torch.allclose(first["logits"], second["logits"])


def test_every_vlm_family_tiny_registration_accepts_real_inputs() -> None:
    tiny_arches = [arch for arch in list_local_arches() if arch.endswith("_tiny")]
    images = torch.randn(1, 3, 16, 16)
    input_ids = torch.tensor([[1, 2, 3, 4]])

    assert len(tiny_arches) == 70
    for arch_id in tiny_arches:
        model = build_local_model(
            arch_id,
            image_size=16,
            vocab_size=32,
            seq_len=4,
            embed_dim=32,
            num_classes=3,
            width_mult=0.25,
            dropout=0.0,
        ).eval()
        with torch.no_grad():
            output = model(images=images, input_ids=input_ids)
        assert output["image_embed"].shape[0] == 1
        assert output["text_embed"].shape[0] == 1
        assert output["logits"].shape[0] == 1
        assert torch.isfinite(output["logits"]).all()


def test_vlm_fidelity_distinguishes_representatives_from_shared_labels() -> None:
    assert (
        fidelity_for_artifact("dlhub/multimodal/vlm/clip.py")
        is FidelityLevel.COMPACT
    )
    assert (
        fidelity_for_artifact("dlhub/multimodal/vlm/llava.py")
        is FidelityLevel.BASELINE_ALIAS
    )
