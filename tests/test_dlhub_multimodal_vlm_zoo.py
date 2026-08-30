import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in VLM zoo contract: {type(x)!r}")


def test_vlm_zoo_lists_20_families_3_variants() -> None:
    from dlhub.multimodal.vlm_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 60
    assert "vlm:clip_tiny" in arches
    assert "vlm:blip_small" in arches
    assert "vlm:flamingo_base" in arches
    assert "vlm:blip2_tiny" in arches
    assert "vlm:llava_small" in arches
    assert "vlm:kosmos2_base" in arches
    assert "vlm:simvlm_tiny" in arches
    assert "vlm:lit_small" in arches
    assert "vlm:pali_base" in arches
    assert "vlm:qwen_vl_small" in arches
    assert "vlm:cogvlm_base" in arches


@pytest.mark.parametrize(
    "arch_id,expect_generated",
    [
        ("vlm:clip_tiny", False),
        ("vlm:simvlm_tiny", True),
        ("vlm:blip_tiny", True),
        ("vlm:pali_tiny", True),
        ("vlm:blip2_tiny", True),
        ("vlm:llava_tiny", True),
        ("vlm:qwen_vl_tiny", True),
    ],
)
def test_vlm_zoo_build_and_forward_contract(arch_id: str, expect_generated: bool) -> None:
    from dlhub.multimodal.vlm_zoo import build_local_model

    model = build_local_model(
        arch_id,
        image_size=32,
        vocab_size=128,
        seq_len=16,
        embed_dim=64,
        num_classes=8,
        width_mult=0.5,
        dropout=0.0,
    )
    out = model.forward(batch_size=2)
    assert isinstance(out, dict)
    assert "image_embed" in out and "text_embed" in out and "logits" in out
    assert tuple(out["image_embed"].shape) == (2, 64)
    assert tuple(out["text_embed"].shape) == (2, 64)
    assert out["logits"].ndim == 2
    assert int(out["logits"].shape[0]) == 2
    if expect_generated:
        assert "generated_tokens" in out
        assert "token_logits" in out
        assert tuple(out["generated_tokens"].shape) == (2, 16)
        assert tuple(out["token_logits"].shape) == (2, 16, 128)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
