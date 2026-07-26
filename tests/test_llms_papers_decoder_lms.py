import pytest

torch = pytest.importorskip("torch")


def test_llama_uses_rmsnorm_rope_and_swiglu() -> None:
    from Llms.llama import LLaMAConfig, LLaMAModel

    model = LLaMAModel(
        LLaMAConfig(
            vocab_size=64,
            max_seq_len=8,
            dim=32,
            n_heads=4,
            n_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    block = model.layers[0]

    assert model.norm.__class__.__name__ == "RMSNorm"
    assert block.attention.use_rope is True
    assert block.feed_forward.activation_name == "swiglu"
    assert not hasattr(model, "position_embeddings")

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids)
    assert tuple(logits.shape) == (2, 8, 64)

def test_llama_forward_with_cache_matches_full_decode() -> None:
    from Llms.llama import LLaMAConfig, LLaMAModel

    model = LLaMAModel(
        LLaMAConfig(
            vocab_size=64,
            max_seq_len=8,
            dim=32,
            n_heads=4,
            n_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    model.eval()
    input_ids = torch.randint(0, 64, (2, 6), dtype=torch.long)

    with torch.no_grad():
        full_logits = model(input_ids)
        _, cache = model.forward_with_cache(input_ids[:, :4])
        cached_logits, next_cache = model.forward_with_cache(
            input_ids[:, 4:],
            past_key_values=cache,
        )

    assert len(cache) == len(model.layers)
    assert len(next_cache) == len(model.layers)
    assert torch.allclose(cached_logits, full_logits[:, 4:], atol=1e-5, rtol=1e-5)

def test_bloom_uses_alibi_and_embedding_layernorm() -> None:
    from Llms.bloom import BloomConfig, BloomModel

    model = BloomModel(
        BloomConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    block = model.h[0]

    assert hasattr(model, "word_embeddings_layernorm")
    assert block.self_attention.use_alibi is True
    assert block.mlp.activation_name == "gelu"

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids)
    assert tuple(logits.shape) == (2, 8, 64)

def test_bloom_forward_with_cache_matches_full_decode() -> None:
    from Llms.bloom import BloomConfig, BloomModel

    model = BloomModel(
        BloomConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    model.eval()
    input_ids = torch.randint(0, 64, (2, 6), dtype=torch.long)

    with torch.no_grad():
        full_logits = model(input_ids)
        _, cache = model.forward_with_cache(input_ids[:, :4])
        cached_logits, next_cache = model.forward_with_cache(
            input_ids[:, 4:],
            past_key_values=cache,
        )

    assert len(cache) == len(model.h)
    assert len(next_cache) == len(model.h)
    assert torch.allclose(cached_logits, full_logits[:, 4:], atol=1e-5, rtol=1e-5)

def test_gpt_neox_uses_partial_rotary_and_parallel_residual() -> None:
    from Llms.gpt_neox import GPTNeoXConfig, GPTNeoXModel

    model = GPTNeoXModel(
        GPTNeoXConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            rotary_pct=0.25,
            dropout=0.0,
        )
    )
    block = model.layers[0]

    assert block.parallel_residual is True
    assert block.attention.rotary_ndims == 2

    hidden = torch.randn(2, 8, 32)
    attention_mask = torch.ones(2, 8, dtype=torch.float32)
    with torch.no_grad():
        attn_in = block.input_layernorm(hidden)
        mlp_in = block.post_attention_layernorm(hidden)
        expected = hidden + block.attention(attn_in, attention_mask) + block.mlp(mlp_in)
        actual = block(hidden, attention_mask)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids)
    assert tuple(logits.shape) == (2, 8, 64)

def test_gpt_neox_forward_with_cache_matches_full_decode() -> None:
    from Llms.gpt_neox import GPTNeoXConfig, GPTNeoXModel

    model = GPTNeoXModel(
        GPTNeoXConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            rotary_pct=0.25,
            dropout=0.0,
        )
    )
    model.eval()
    input_ids = torch.randint(0, 64, (2, 6), dtype=torch.long)

    with torch.no_grad():
        full_logits = model(input_ids)
        _, cache = model.forward_with_cache(input_ids[:, :4])
        cached_logits, next_cache = model.forward_with_cache(
            input_ids[:, 4:],
            past_key_values=cache,
        )

    assert len(cache) == len(model.layers)
    assert len(next_cache) == len(model.layers)
    assert torch.allclose(cached_logits, full_logits[:, 4:], atol=1e-5, rtol=1e-5)

def test_lora_linear_zero_init_and_merge_roundtrip() -> None:
    from Llms.lora import LoRAConfig, LoRALinear

    base = torch.nn.Linear(16, 12, bias=False)
    layer = LoRALinear.from_linear(base, LoRAConfig(rank=4, alpha=8.0, dropout=0.0))
    x = torch.randn(3, 16)

    with torch.no_grad():
        expected = base(x)
        actual = layer(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert layer.base.weight.requires_grad is False
    assert layer.lora_a.requires_grad is True
    assert layer.lora_b.requires_grad is True

    with torch.no_grad():
        layer.lora_b.fill_(0.25)
        unmerged = layer(x)
        layer.merge()
        merged = layer(x)
        layer.unmerge()
        roundtrip = layer(x)

    assert torch.allclose(unmerged, merged, atol=1e-5, rtol=1e-5)
    assert torch.allclose(unmerged, roundtrip, atol=1e-5, rtol=1e-5)

def test_llama_adapter_zero_gate_preserves_base_logits_initially() -> None:
    from Llms.llama import LLaMAConfig, LLaMAModel
    from Llms.llama_adapter import LLaMAAdapterConfig, LLaMAAdapterModel

    base = LLaMAModel(
        LLaMAConfig(
            vocab_size=64,
            max_seq_len=8,
            dim=32,
            n_heads=4,
            n_layers=4,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    adapter = LLaMAAdapterModel(
        base,
        LLaMAAdapterConfig(prompt_length=3, adapter_layers=2),
    )
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)

    with torch.no_grad():
        base_logits = base(input_ids)
        adapter_logits = adapter(input_ids)

    assert adapter.target_layer_indices == [2, 3]
    assert torch.allclose(base_logits, adapter_logits, atol=1e-6, rtol=1e-6)

    with torch.no_grad():
        for gate in adapter.gates:
            gate.gate.fill_(1.0)
        changed_logits = adapter(input_ids)

    assert tuple(changed_logits.shape) == tuple(base_logits.shape)
    assert not torch.allclose(base_logits, changed_logits)

def test_palm_uses_mqa_parallel_block_and_weight_tying() -> None:
    from Llms.palm import PaLMConfig, PaLMModel

    model = PaLMModel(
        PaLMConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    block = model.layers[0]

    assert block.parallel_residual is True
    assert block.attention.use_rope is True
    assert block.attention.multi_query is True
    assert block.attention.k_proj.out_features == block.attention.head_dim
    assert block.attention.v_proj.out_features == block.attention.head_dim
    assert block.feed_forward.activation_name == "swiglu"
    assert block.attention.q_proj.bias is None
    assert block.attention.o_proj.bias is None
    assert model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert not hasattr(model, "position_embeddings")

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids)
    assert tuple(logits.shape) == (2, 8, 64)

def test_palm_forward_with_cache_matches_full_decode_and_keeps_shared_kv() -> None:
    from Llms.palm import PaLMConfig, PaLMModel

    model = PaLMModel(
        PaLMConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    model.eval()
    input_ids = torch.randint(0, 64, (2, 6), dtype=torch.long)

    with torch.no_grad():
        full_logits = model(input_ids)
        _, cache = model.forward_with_cache(input_ids[:, :4])
        cached_logits, next_cache = model.forward_with_cache(
            input_ids[:, 4:],
            past_key_values=cache,
        )

    assert len(cache) == len(model.layers)
    assert len(next_cache) == len(model.layers)
    assert cache[0][0].shape[1] == 1
    assert cache[0][1].shape[1] == 1
    assert torch.allclose(cached_logits, full_logits[:, 4:], atol=1e-5, rtol=1e-5)

def test_lamda_uses_relative_attention_gated_gelu_and_grounding_tools() -> None:
    from Llms.lamda import LaMDAConfig, LaMDAModel, LaMDAToolset

    toolset = LaMDAToolset()
    assert toolset.tool_names == ["calculator", "translator", "retrieval"]
    assert toolset.run("calculator", "135+7721") == ["7856"]
    assert toolset.route_query("hello in French") == "translator"
    assert toolset.route_query("How old is Rafael Nadal?") == "retrieval"

    model = LaMDAModel(
        LaMDAConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    block = model.layers[0]

    assert not hasattr(model, "encoder")
    assert block.self_attention.relative_attention_bias is not None
    assert block.feed_forward.activation_name == "gated_gelu"
    assert model.default_tool_order == ("calculator", "translator", "retrieval")

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids)
    assert tuple(logits.shape) == (2, 8, 64)

def test_lamda_quality_metrics_follow_ssi_intuition() -> None:
    from Llms.lamda import LaMDADialogAgent

    agent = LaMDADialogAgent()
    generic = agent.score_quality(
        "User: I love Eurovision.",
        "Me too.",
    )
    specific = agent.score_quality(
        "User: I love Eurovision.",
        "Me too. I love Eurovision songs.",
    )

    assert generic.sensibleness == 1.0
    assert generic.specificity == 0.0
    assert specific.specificity == 1.0
    assert specific.ssi > generic.ssi

def test_lamda_dialog_agent_filters_unsafe_candidates_and_prefers_grounded_tool_answers() -> None:
    from Llms.lamda import LaMDADialogAgent

    agent = LaMDADialogAgent()
    result = agent.respond(
        "135+7721",
        candidate_responses=[
            "I am not sure.",
            "You should buy illegal drugs online.",
        ],
    )

    assert result.text.startswith("That would be 7856.")
    assert result.safety == 1.0
    assert result.groundedness == 1.0
    assert "calculator" in result.citations[0]

def test_pythia_wraps_gpt_neox_with_checkpoint_schedule_and_suite_metadata() -> None:
    from Llms.pythia import PythiaCheckpointSchedule, PythiaConfig, PythiaModel

    schedule = PythiaCheckpointSchedule()
    assert schedule.steps[:11] == [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    assert 1000 in schedule.steps
    assert len(schedule.steps) == 154

    model = PythiaModel(
        PythiaConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
            deduped=True,
        )
    )
    assert model.base_model.__class__.__name__ == "GPTNeoXModel"
    assert model.uses_flash_attention is True
    assert model.data_order_is_reconstructable is True
    assert model.config.deduped is True
    assert model.base_model.layers[0].parallel_residual is True
    assert model.base_model.layers[0].attention.rotary_ndims == 2
    assert model.base_model.embed_in.weight.data_ptr() != model.base_model.embed_out.weight.data_ptr()

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids)
    assert tuple(logits.shape) == (2, 8, 64)

def test_fed_uses_multi_query_attention_and_incremental_cache_matches_full_decode() -> None:
    from Llms.fed import FEDConfig, FEDModel

    model = FEDModel(
        FEDConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    block = model.layers[0]

    assert block.attention.multi_query is True
    assert block.attention.k_proj.out_features == block.attention.head_dim
    assert block.attention.v_proj.out_features == block.attention.head_dim
    assert (
        block.attention.cache_elements_per_token()
        < block.attention.standard_mha_cache_elements_per_token()
    )

    input_ids = torch.randint(0, 64, (2, 5), dtype=torch.long)
    with torch.no_grad():
        full_logits = model(input_ids)
        _, cache = model(input_ids[:, :4], use_cache=True)
        next_logits, next_cache = model(
            input_ids[:, 4:],
            past_key_values=cache,
            use_cache=True,
        )

    assert tuple(full_logits.shape) == (2, 5, 64)
    assert len(next_cache) == 2
    assert next_cache[0][0].shape == (2, 5, 8)
    assert torch.allclose(next_logits[:, -1], full_logits[:, -1], atol=1e-5, rtol=1e-5)
