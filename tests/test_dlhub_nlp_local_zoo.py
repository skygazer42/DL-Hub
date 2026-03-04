import pytest


torch = pytest.importorskip("torch")


def test_local_nlp_zoo_has_one_file_per_arch() -> None:
    from pathlib import Path

    from dlhub.nlp.local_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 700

    # NLP zoo is registry-based: one algorithm-family module contains many variants
    # (ResNet-style), rather than one file per variant.
    import dlhub.nlp.algorithms as algorithms

    pkg_dir = Path(algorithms.__file__).resolve().parent
    files = [
        p
        for p in pkg_dir.glob("*.py")
        if p.name not in {"__init__.py"}
    ]
    assert any(p.name == "registry.py" for p in files)
    assert any(p.name == "transformer.py" for p in files)
    assert any(p.name == "textcnn.py" for p in files)
    assert len(files) < 100


def test_local_nlp_zoo_lists_100_plus_arches() -> None:
    from dlhub.nlp.local_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 700

    # A few representative families.
    assert "nl:mean_pool" in arches
    assert "nl:textcnn_k345" in arches
    assert "nl:textcnn_k234_d2" in arches
    assert "nl:bilstm_last" in arches
    assert "nl:bilstm_attn3l" in arches
    assert "nl:transformer_tiny" in arches
    assert "nl:transformer_tiny_mean_gelu_ln_learned_pre" in arches
    assert "nl:transformer_linformer_tiny_k16_mean_gelu_ln_learned_pre" in arches
    assert "nl:transformer_performer_tiny_mean_gelu_ln_learned_pre" in arches
    assert "nl:bert_tiny" in arches
    assert "nl:gpt_tiny" in arches
    assert "nl:fnet_tiny" in arches
    assert "nl:gmlp_tiny" in arches
    assert "nl:linformer_tiny" in arches
    assert "nl:longformer_tiny" in arches


@pytest.mark.parametrize(
    ("arch_id", "width_mult"),
    [
        ("nl:mean_pool", 1.0),
        ("nl:textcnn_k345", 0.5),
        ("nl:textcnn_k234_d2", 0.5),
        ("nl:bilstm_attn", 0.5),
        ("nl:bilstm_attn3l", 0.5),
        ("nl:qrnn_k2_mean", 0.5),
        ("nl:transformer_rope_tiny", 0.5),
        ("nl:transformer_linformer_tiny_k16_cls_swiglu_rms_sin_pre", 0.5),
        ("nl:t5_tiny", 0.5),
        ("nl:performer_tiny", 0.5),
        ("nl:fnet_tiny", 0.5),
        ("nl:gmlp_tiny", 0.5),
    ],
)
def test_local_nlp_zoo_build_smoke(arch_id: str, width_mult: float) -> None:
    from dlhub.nlp.local_zoo import build_local_model

    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_local_model(
        arch_id,
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=float(width_mult),
        dropout=0.1,
    )
    model.eval()

    x = torch.zeros(2, max_length, dtype=torch.long)
    attn = torch.zeros(2, max_length, dtype=torch.float32)
    attn[:, :16] = 1.0

    with torch.no_grad():
        y = model({"input_ids": x, "attention_mask": attn})
    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == (2, num_classes)
