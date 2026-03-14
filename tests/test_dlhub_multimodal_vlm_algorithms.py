import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported VLM output type: {type(x)!r}")


@pytest.mark.parametrize(
    "builder_name,kwargs,expect_generated",
    [
        (
            "build_clip_vlm",
            {
                "image_size": 32,
                "vocab_size": 128,
                "seq_len": 16,
                "embed_dim": 64,
                "num_classes": 8,
                "variant": "clip_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
            False,
        ),
        (
            "build_blip_vlm",
            {
                "image_size": 32,
                "vocab_size": 128,
                "seq_len": 16,
                "embed_dim": 64,
                "num_classes": 8,
                "variant": "blip_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
            True,
        ),
        (
            "build_blip2_vlm",
            {
                "image_size": 32,
                "vocab_size": 128,
                "seq_len": 16,
                "embed_dim": 64,
                "num_classes": 8,
                "variant": "blip2_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
            True,
        ),
        (
            "build_llava_vlm",
            {
                "image_size": 32,
                "vocab_size": 128,
                "seq_len": 16,
                "embed_dim": 64,
                "num_classes": 8,
                "variant": "llava_tiny",
                "width_mult": 0.5,
                "dropout": 0.0,
            },
            True,
        ),
    ],
)
def test_vlm_algorithms_forward_smoke(
    builder_name: str,
    kwargs: dict,
    expect_generated: bool,
) -> None:
    import dlhub.multimodal.vlm as vlm

    build = getattr(vlm, builder_name)
    model = build(**kwargs)
    out = model.forward(batch_size=2)
    assert isinstance(out, dict)
    assert "image_embed" in out
    assert "text_embed" in out
    assert "logits" in out
    assert tuple(out["image_embed"].shape) == (2, 64)
    assert tuple(out["text_embed"].shape) == (2, 64)
    assert out["logits"].ndim == 2
    assert int(out["logits"].shape[0]) == 2
    if expect_generated:
        assert tuple(out["generated_tokens"].shape) == (2, 16)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
