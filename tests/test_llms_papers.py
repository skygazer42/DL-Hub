import pytest
import importlib
from pathlib import Path

torch = pytest.importorskip("torch")


def test_llms_package_exports_paper_modules() -> None:
    import Llms
    from Llms import (
        ai_bubbles,
        anthropic,
        bard,
        blip2,
        blip,
        bloom,
        chain_of_thought,
        chinchilla,
        dolly,
        fed,
        flamingo,
        flan_t5,
        gpipe,
        gpt4all,
        gpt4all_j,
        gpt_neox,
        helm,
        imagen,
        instructblip,
        instructgpt,
        lamda,
        llm_survey,
        llm_timeline,
        llama,
        llama_adapter,
        lora,
        megatron,
        minigpt4,
        mtf,
        openassistant,
        oasst1,
        parameter_server,
        palm,
        pile,
        pathways,
        prompt_engineering_guide,
        pythia,
        red_pajama,
        scienceqa,
        segment_anything,
        self_instruct,
        starcoder,
        t5,
        the_stack,
        ul2,
        vilt,
        zero,
    )

    assert hasattr(Llms, "llama")
    assert hasattr(Llms, "bloom")
    assert hasattr(Llms, "gpt_neox")
    assert hasattr(Llms, "lora")
    assert hasattr(Llms, "llama_adapter")
    assert hasattr(Llms, "t5")
    assert hasattr(Llms, "flan_t5")
    assert hasattr(Llms, "ul2")
    assert hasattr(Llms, "ai_bubbles")
    assert hasattr(Llms, "bard")
    assert hasattr(Llms, "blip2")
    assert hasattr(Llms, "blip")
    assert hasattr(Llms, "flamingo")
    assert hasattr(Llms, "minigpt4")
    assert hasattr(Llms, "palm")
    assert hasattr(Llms, "lamda")
    assert hasattr(Llms, "megatron")
    assert hasattr(Llms, "instructgpt")
    assert hasattr(Llms, "instructblip")
    assert hasattr(Llms, "llm_survey")
    assert hasattr(Llms, "llm_timeline")
    assert hasattr(Llms, "self_instruct")
    assert hasattr(Llms, "pythia")
    assert hasattr(Llms, "anthropic")
    assert hasattr(Llms, "imagen")
    assert hasattr(Llms, "vilt")
    assert hasattr(Llms, "chinchilla")
    assert hasattr(Llms, "gpipe")
    assert hasattr(Llms, "pathways")
    assert hasattr(Llms, "zero")
    assert hasattr(Llms, "parameter_server")
    assert hasattr(Llms, "fed")
    assert hasattr(Llms, "helm")
    assert hasattr(Llms, "pile")
    assert hasattr(Llms, "openassistant")
    assert hasattr(Llms, "oasst1")
    assert hasattr(Llms, "the_stack")
    assert hasattr(Llms, "prompt_engineering_guide")
    assert hasattr(Llms, "red_pajama")
    assert hasattr(Llms, "scienceqa")
    assert hasattr(Llms, "dolly")
    assert hasattr(Llms, "gpt4all")
    assert hasattr(Llms, "segment_anything")
    assert hasattr(Llms, "starcoder")
    assert hasattr(Llms, "chain_of_thought")
    assert hasattr(Llms, "gpt4all_j")
    assert hasattr(Llms, "mtf")

    assert hasattr(llama, "LLaMAConfig")
    assert hasattr(llama, "LLaMAModel")
    assert hasattr(bloom, "BloomConfig")
    assert hasattr(bloom, "BloomModel")
    assert hasattr(gpt_neox, "GPTNeoXConfig")
    assert hasattr(gpt_neox, "GPTNeoXModel")
    assert hasattr(lora, "LoRAConfig")
    assert hasattr(lora, "LoRALinear")
    assert hasattr(llama_adapter, "LLaMAAdapterConfig")
    assert hasattr(llama_adapter, "LLaMAAdapterModel")
    assert hasattr(t5, "T5Config")
    assert hasattr(t5, "T5Model")
    assert hasattr(flan_t5, "FlanT5Config")
    assert hasattr(flan_t5, "FlanT5Model")
    assert hasattr(ul2, "UL2Config")
    assert hasattr(ul2, "UL2Model")
    assert hasattr(ai_bubbles, "AIBubbleEntry")
    assert hasattr(ai_bubbles, "AIBubblesRegistry")
    assert hasattr(bard, "BardConfig")
    assert hasattr(bard, "BardSession")
    assert hasattr(blip2, "BLIP2Config")
    assert hasattr(blip2, "BLIP2Model")
    assert hasattr(blip, "BLIPConfig")
    assert hasattr(blip, "BLIPModel")
    assert hasattr(flamingo, "FlamingoConfig")
    assert hasattr(flamingo, "FlamingoModel")
    assert hasattr(minigpt4, "MiniGPT4Config")
    assert hasattr(minigpt4, "MiniGPT4Model")
    assert hasattr(palm, "PaLMConfig")
    assert hasattr(palm, "PaLMModel")
    assert hasattr(lamda, "LaMDAConfig")
    assert hasattr(lamda, "LaMDAModel")
    assert hasattr(megatron, "MegatronConfig")
    assert hasattr(megatron, "MegatronModel")
    assert hasattr(instructgpt, "InstructGPTConfig")
    assert hasattr(instructgpt, "InstructGPTModel")
    assert hasattr(instructblip, "InstructBLIPConfig")
    assert hasattr(instructblip, "InstructBLIPModel")
    assert hasattr(llm_survey, "LLMSurveyGuide")
    assert hasattr(llm_timeline, "LLMTimeline")
    assert hasattr(self_instruct, "SelfInstructConfig")
    assert hasattr(self_instruct, "SelfInstructModel")
    assert hasattr(pythia, "PythiaConfig")
    assert hasattr(pythia, "PythiaModel")
    assert hasattr(anthropic, "AnthropicConfig")
    assert hasattr(anthropic, "AnthropicModel")
    assert hasattr(imagen, "ImagenConfig")
    assert hasattr(imagen, "ImagenModel")
    assert hasattr(vilt, "ViLTConfig")
    assert hasattr(vilt, "ViLTModel")
    assert hasattr(chinchilla, "ChinchillaConfig")
    assert hasattr(chinchilla, "ChinchillaPlanner")
    assert hasattr(gpipe, "GPipeConfig")
    assert hasattr(gpipe, "GPipeSequential")
    assert hasattr(pathways, "PathwaysConfig")
    assert hasattr(pathways, "PathwaysRuntime")
    assert hasattr(zero, "ZeROConfig")
    assert hasattr(zero, "ZeROEngine")
    assert hasattr(parameter_server, "ParameterServerConfig")
    assert hasattr(parameter_server, "ParameterServer")
    assert hasattr(fed, "FEDConfig")
    assert hasattr(fed, "FEDModel")
    assert hasattr(helm, "HELMScenario")
    assert hasattr(helm, "HELMEvaluator")
    assert hasattr(pile, "PileConfig")
    assert hasattr(pile, "PileMixture")
    assert hasattr(openassistant, "OpenAssistantConfig")
    assert hasattr(openassistant, "OpenAssistantDataset")
    assert hasattr(oasst1, "OASST1Config")
    assert hasattr(oasst1, "OASST1Dataset")
    assert hasattr(prompt_engineering_guide, "PromptEngineeringConfig")
    assert hasattr(prompt_engineering_guide, "PromptEngineeringGuide")
    assert hasattr(red_pajama, "RedPajamaConfig")
    assert hasattr(red_pajama, "RedPajamaDataset")
    assert hasattr(the_stack, "TheStackConfig")
    assert hasattr(the_stack, "TheStackDataset")
    assert hasattr(scienceqa, "ScienceQAConfig")
    assert hasattr(scienceqa, "ScienceQAModel")
    assert hasattr(dolly, "DollyConfig")
    assert hasattr(dolly, "DollyModel")
    assert hasattr(gpt4all, "GPT4AllConfig")
    assert hasattr(gpt4all, "GPT4AllModel")
    assert hasattr(segment_anything, "SAMConfig")
    assert hasattr(segment_anything, "SegmentAnythingModel")
    assert hasattr(starcoder, "StarCoderConfig")
    assert hasattr(starcoder, "StarCoderModel")
    assert hasattr(chain_of_thought, "ChainOfThoughtConfig")
    assert hasattr(chain_of_thought, "ChainOfThoughtReasoner")
    assert hasattr(gpt4all_j, "GPT4AllJConfig")
    assert hasattr(gpt4all_j, "GPT4AllJModel")
    assert hasattr(mtf, "MTFConfig")
    assert hasattr(mtf, "MTFModel")


def test_llms_title_case_alias_modules_exist() -> None:
    assert importlib.import_module("Llms.LLaMA").LLaMAModel is not None
    assert importlib.import_module("Llms.BLOOM").BloomModel is not None
    assert importlib.import_module("Llms.GPT_NeoX").GPTNeoXModel is not None
    assert importlib.import_module("Llms.LoRA").LoRALinear is not None
    assert importlib.import_module("Llms.LLaMA_Adapter").LLaMAAdapterModel is not None
    assert importlib.import_module("Llms.T5").T5Model is not None
    assert importlib.import_module("Llms.Flan_T5").FlanT5Model is not None
    assert importlib.import_module("Llms.UL2").UL2Model is not None
    assert importlib.import_module("Llms.AI_Bubbles").AIBubblesRegistry is not None
    assert importlib.import_module("Llms.Bard").BardSession is not None
    assert importlib.import_module("Llms.BLIP2").BLIP2Model is not None
    assert importlib.import_module("Llms.BLIP").BLIPModel is not None
    assert importlib.import_module("Llms.Flamingo").FlamingoModel is not None
    assert importlib.import_module("Llms.MiniGPT4").MiniGPT4Model is not None
    assert importlib.import_module("Llms.PaLM").PaLMModel is not None
    assert importlib.import_module("Llms.LaMDA").LaMDAModel is not None
    assert importlib.import_module("Llms.Megatron").MegatronModel is not None
    assert importlib.import_module("Llms.InstructGPT").InstructGPTModel is not None
    assert importlib.import_module("Llms.InstructBLIP").InstructBLIPModel is not None
    assert importlib.import_module("Llms.LLM_Survey").LLMSurveyGuide is not None
    assert importlib.import_module("Llms.LLM_Timeline").LLMTimeline is not None
    assert importlib.import_module("Llms.Self_Instruct").SelfInstructModel is not None
    assert importlib.import_module("Llms.Pythia").PythiaModel is not None
    assert importlib.import_module("Llms.Anthropic").AnthropicModel is not None
    assert importlib.import_module("Llms.Imagen").ImagenModel is not None
    assert importlib.import_module("Llms.ViLT").ViLTModel is not None
    assert importlib.import_module("Llms.Chinchilla").ChinchillaPlanner is not None
    assert importlib.import_module("Llms.GPipe").GPipeSequential is not None
    assert importlib.import_module("Llms.Pathways").PathwaysRuntime is not None
    assert importlib.import_module("Llms.ZeRO").ZeROEngine is not None
    assert importlib.import_module("Llms.Parameter_Server").ParameterServer is not None
    assert importlib.import_module("Llms.FED").FEDModel is not None
    assert importlib.import_module("Llms.HELM").HELMEvaluator is not None
    assert importlib.import_module("Llms.OpenAssistant").OpenAssistantDataset is not None
    assert importlib.import_module("Llms.OASST1").OASST1Dataset is not None
    assert importlib.import_module("Llms.Pile").PileMixture is not None
    assert importlib.import_module("Llms.Prompt_Engineering_Guide").PromptEngineeringGuide is not None
    assert importlib.import_module("Llms.RedPajama").RedPajamaDataset is not None
    assert importlib.import_module("Llms.The_Stack").TheStackDataset is not None
    assert importlib.import_module("Llms.ScienceQA").ScienceQAModel is not None
    assert importlib.import_module("Llms.Dolly").DollyModel is not None
    assert importlib.import_module("Llms.GPT4All").GPT4AllModel is not None
    assert importlib.import_module("Llms.Segment_Anything").SegmentAnythingModel is not None
    assert importlib.import_module("Llms.StarCoder").StarCoderModel is not None
    assert importlib.import_module("Llms.Chain_of_Thought").ChainOfThoughtReasoner is not None
    assert importlib.import_module("Llms.GPT4All_J").GPT4AllJModel is not None
    assert importlib.import_module("Llms.MTF").MTFModel is not None


def test_llms_resource_registry_covers_all_llm_resource_files() -> None:
    from Llms.resource_registry import LLM_RESOURCE_INDEX

    resource_files = {
        path.name
        for path in Path("resources/pdfs/llms").iterdir()
        if path.is_file()
    }
    assert set(LLM_RESOURCE_INDEX) == resource_files


def test_llms_resource_registry_imports_mapped_modules_and_marks_reference_only_files() -> None:
    from Llms.resource_registry import LLM_RESOURCE_INDEX

    reference_only = {
        name
        for name, entry in LLM_RESOURCE_INDEX.items()
        if entry.status == "reference_only"
    }
    assert reference_only == {"dataset (3-5).pdf", "大模型.md"}

    for entry in LLM_RESOURCE_INDEX.values():
        for module_name in entry.module_names:
            mod = importlib.import_module(f"Llms.{module_name}")
            assert mod is not None


def test_llms_resource_registry_handles_filename_mismatches_and_duplicates() -> None:
    from Llms.resource_registry import LLM_RESOURCE_INDEX

    assert LLM_RESOURCE_INDEX["Chinchilia .pdf"].module_names == ("chinchilla",)
    assert LLM_RESOURCE_INDEX["mingpt4.pdf"].module_names == ("minigpt4",)
    assert LLM_RESOURCE_INDEX["timeline1.pdf"].module_names == ("llm_timeline",)
    assert LLM_RESOURCE_INDEX["PaLM (Scaling Language Modeling with Pathways).md"].module_names == (
        "palm",
    )
    assert LLM_RESOURCE_INDEX["多模态统一框架之BLIP系列工作.pdf"].module_names == (
        "blip",
        "blip2",
        "instructblip",
    )


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


def test_t5_uses_shared_embeddings_and_relative_bias() -> None:
    from Llms.t5 import T5Config, T5Model

    model = T5Model(
        T5Config(
            vocab_size=64,
            max_seq_len=8,
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=64,
            dropout=0.0,
        )
    )
    assert model.shared is model.encoder.embed_tokens
    assert model.shared is model.decoder.embed_tokens
    assert hasattr(model.encoder.blocks[0].self_attention, "relative_attention_bias")
    assert hasattr(model.decoder.blocks[0].self_attention, "relative_attention_bias")
    assert not hasattr(model, "position_embeddings")

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    decoder_input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    assert tuple(logits.shape) == (2, 8, 64)


def test_t5_encoder_relative_position_bias_is_bidirectional() -> None:
    from Llms.t5 import T5Config, T5Model

    model = T5Model(
        T5Config(
            vocab_size=64,
            max_seq_len=8,
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=64,
            dropout=0.0,
        )
    )
    encoder_bias = model.encoder.blocks[0].self_attention.relative_attention_bias
    decoder_bias = model.decoder.blocks[0].self_attention.relative_attention_bias
    rel = torch.tensor([[1, 0, -1]], dtype=torch.long)

    assert encoder_bias.bidirectional is True
    assert decoder_bias.bidirectional is False
    buckets = encoder_bias._relative_position_bucket(rel)
    assert len(set(buckets.flatten().tolist())) == 3


def test_flan_t5_wraps_t5_and_formats_instruction_prompt() -> None:
    from Llms.flan_t5 import FlanT5Config, FlanT5Model, format_instruction_prompt

    prompt = format_instruction_prompt("Answer the question.", "What is 2+2?")
    assert prompt.startswith("Answer the question.")
    assert "What is 2+2?" in prompt

    model = FlanT5Model(
        FlanT5Config(
            vocab_size=64,
            max_seq_len=8,
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=64,
            dropout=0.0,
        )
    )
    assert model.base_model.__class__.__name__ == "T5Model"

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    decoder_input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    assert tuple(logits.shape) == (2, 8, 64)


def test_ul2_exposes_modes_and_t5_compatible_forward() -> None:
    from Llms.ul2 import UL2Config, UL2Model, UL2Objective

    objective = UL2Objective()
    assert objective.mode_to_tag["R"] == "[NLU]"
    assert objective.mode_to_tag["S"] == "[S2S]"
    assert objective.mode_to_tag["X"] == "[NLG]"
    assert objective.format_with_mode("translate this", mode="S").startswith("[S2S]")

    model = UL2Model(
        UL2Config(
            vocab_size=64,
            max_seq_len=8,
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=64,
            dropout=0.0,
        )
    )
    assert model.base_model.__class__.__name__ == "T5Model"

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    decoder_input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, mode="R")
    assert tuple(logits.shape) == (2, 8, 64)


def test_ul2_objective_builds_mode_specific_noising_plans() -> None:
    from Llms.ul2 import UL2Objective

    objective = UL2Objective()
    tokens = list(range(10))

    r = objective.apply_mode(tokens, mode="R", sentinel_start=100)
    s = objective.apply_mode(tokens, mode="S", sentinel_start=100)
    x = objective.apply_mode(tokens, mode="X", sentinel_start=100)

    assert r.plan.objective_name == "regular_span_corruption"
    assert r.plan.num_masked_tokens == 2
    assert r.plan.mean_noise_span_length == 3.0
    assert any(token >= 100 for token in r.corrupted_input)
    assert r.target[0] >= 100

    assert s.plan.objective_name == "prefix_lm"
    assert s.plan.prefix_length == 5
    assert s.corrupted_input == tokens[:5]
    assert s.target == tokens[5:]

    assert x.plan.objective_name == "extreme_span_corruption"
    assert x.plan.num_masked_tokens == 5
    assert x.plan.mean_noise_span_length == 32.0
    assert x.plan.num_masked_tokens > r.plan.num_masked_tokens
    assert any(token >= 100 for token in x.corrupted_input)


def test_blip2_has_qformer_and_frozen_backbones() -> None:
    from Llms.blip2 import BLIP2Config, BLIP2Model

    model = BLIP2Model(
        BLIP2Config(
            vocab_size=64,
            max_seq_len=8,
            llm_dim=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            image_feat_dim=24,
            num_query_tokens=4,
            qformer_hidden_size=24,
            dropout=0.0,
        )
    )
    assert tuple(model.query_tokens.shape) == (4, 24)
    assert all(not p.requires_grad for p in model.image_encoder.parameters())
    assert all(not p.requires_grad for p in model.llm.parameters())
    assert model.q_former is not None
    assert model.llm_projection is not None

    image_features = torch.randn(2, 5, 24)
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(image_features=image_features, input_ids=input_ids)
    assert tuple(logits.shape) == (2, 8, 64)


def test_flamingo_resampler_and_zero_gated_cross_attention() -> None:
    from Llms.flamingo import FlamingoConfig, FlamingoModel

    model = FlamingoModel(
        FlamingoConfig(
            vocab_size=64,
            max_seq_len=8,
            llm_dim=32,
            num_heads=4,
            num_layers=4,
            intermediate_size=64,
            image_feat_dim=24,
            resampler_num_latents=4,
            cross_attn_every_n_layers=2,
            dropout=0.0,
        )
    )
    assert model.perceiver_resampler is not None
    assert model.cross_attn_layer_indices == [0, 2]
    assert torch.allclose(model.gated_xattn_layers[0].attn_gate, torch.zeros_like(model.gated_xattn_layers[0].attn_gate))

    image_features = torch.randn(2, 6, 24)
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    with torch.no_grad():
        base_logits = model.llm(input_ids)
        flamingo_logits = model(image_features=image_features, input_ids=input_ids)
    assert torch.allclose(base_logits, flamingo_logits, atol=1e-6, rtol=1e-6)

    with torch.no_grad():
        for layer in model.gated_xattn_layers:
            layer.attn_gate.fill_(1.0)
        changed = model(image_features=image_features, input_ids=input_ids)
    assert tuple(changed.shape) == tuple(base_logits.shape)
    assert not torch.allclose(base_logits, changed)


def test_minigpt4_uses_single_projection_layer_into_frozen_llm() -> None:
    from Llms.minigpt4 import MiniGPT4Config, MiniGPT4Model, format_minigpt4_prompt

    prompt = format_minigpt4_prompt("Describe this image in detail.")
    assert "<Img><ImageFeature></Img>" in prompt
    assert "Describe this image in detail." in prompt

    model = MiniGPT4Model(
        MiniGPT4Config(
            vocab_size=64,
            max_seq_len=8,
            llm_dim=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            image_feat_dim=24,
            visual_token_count=4,
            dropout=0.0,
        )
    )
    assert isinstance(model.vision_projection, torch.nn.Linear)
    assert all(not p.requires_grad for p in model.vision_encoder.parameters())
    assert all(not p.requires_grad for p in model.llm.parameters())

    image_features = torch.randn(2, 4, 24)
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(image_features=image_features, input_ids=input_ids)
    assert tuple(logits.shape) == (2, 8, 64)


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


def test_megatron_uses_tensor_parallel_layers_and_vocab_partitioning() -> None:
    from Llms.megatron import MegatronConfig, MegatronModel

    model = MegatronModel(
        MegatronConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_attention_heads=4,
            num_layers=2,
            intermediate_size=64,
            tensor_model_parallel_size=2,
            tensor_model_parallel_rank=0,
            dropout=0.0,
        )
    )
    block = model.layers[0]

    assert block.attention.num_attention_heads_per_partition == 2
    assert block.attention.query_key_value.is_column_parallel is True
    assert block.mlp.dense_h_to_4h.is_column_parallel is True
    assert block.mlp.dense_4h_to_h.is_row_parallel is True
    assert block.mlp.activation_name == "gelu"
    assert model.word_embeddings.is_vocab_parallel is True
    assert model.word_embeddings.vocab_start_index == 0
    assert model.word_embeddings.vocab_end_index == 32
    assert model.word_embeddings.weight.shape[0] == 32
    assert model.word_embeddings.weight.data_ptr() == model.lm_head.weight.data_ptr()

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids)
    assert tuple(logits.shape) == (2, 8, 64)


def test_megatron_parallel_layers_can_load_and_reconstruct_dense_weights() -> None:
    import torch.nn.functional as F
    from Llms.megatron import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding

    full_col_weight = torch.randn(12, 8)
    full_col_bias = torch.randn(12)
    col0 = ColumnParallelLinear(
        8, 12, tensor_model_parallel_size=2, tensor_model_parallel_rank=0, bias=True
    )
    col1 = ColumnParallelLinear(
        8, 12, tensor_model_parallel_size=2, tensor_model_parallel_rank=1, bias=True
    )
    col0.load_full_parameters(full_col_weight, full_col_bias)
    col1.load_full_parameters(full_col_weight, full_col_bias)
    x = torch.randn(2, 3, 8)
    dense_col = F.linear(x, full_col_weight, full_col_bias)
    parted_col = torch.cat((col0(x), col1(x)), dim=-1)
    gathered_col_weight, gathered_col_bias = ColumnParallelLinear.gather_full_parameters(
        (col0, col1)
    )
    assert torch.allclose(parted_col, dense_col, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gathered_col_weight, full_col_weight)
    assert torch.allclose(gathered_col_bias, full_col_bias)

    full_row_weight = torch.randn(10, 8)
    full_row_bias = torch.randn(10)
    row0 = RowParallelLinear(
        8, 10, tensor_model_parallel_size=2, tensor_model_parallel_rank=0, bias=True
    )
    row1 = RowParallelLinear(
        8, 10, tensor_model_parallel_size=2, tensor_model_parallel_rank=1, bias=True
    )
    row0.load_full_parameters(full_row_weight, full_row_bias)
    row1.load_full_parameters(full_row_weight, full_row_bias)
    dense_row = F.linear(x, full_row_weight, full_row_bias)
    parted_row = row0.forward_partial(x) + row1.forward_partial(x) + full_row_bias
    gathered_row_weight, gathered_row_bias = RowParallelLinear.gather_full_parameters((row0, row1))
    assert torch.allclose(parted_row, dense_row, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gathered_row_weight, full_row_weight)
    assert torch.allclose(gathered_row_bias, full_row_bias)

    full_embed_weight = torch.randn(16, 6)
    emb0 = VocabParallelEmbedding(
        16, 6, tensor_model_parallel_size=2, tensor_model_parallel_rank=0
    )
    emb1 = VocabParallelEmbedding(
        16, 6, tensor_model_parallel_size=2, tensor_model_parallel_rank=1
    )
    emb0.load_full_weight(full_embed_weight)
    emb1.load_full_weight(full_embed_weight)
    token_ids = torch.tensor([[0, 5, 8, 15]], dtype=torch.long)
    dense_embed = F.embedding(token_ids, full_embed_weight)
    parted_embed = emb0(token_ids) + emb1(token_ids)
    gathered_embed = VocabParallelEmbedding.gather_full_weight((emb0, emb1))
    assert torch.allclose(parted_embed, dense_embed, atol=1e-5, rtol=1e-5)
    assert torch.allclose(gathered_embed, full_embed_weight)


def test_instructgpt_exposes_rlhf_stages_reward_model_and_ppo_ptx_objective() -> None:
    from Llms.instructgpt import InstructGPTConfig, InstructGPTModel

    model = InstructGPTModel(
        InstructGPTConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
            kl_coeff=0.02,
            pretraining_mix_coeff=27.8,
        )
    )

    assert model.stage_order == ("sft", "reward_model", "ppo")
    assert model.reward_model.reward_head.out_features == 1
    assert all(not p.requires_grad for p in model.reference_policy.parameters())
    assert model.objective.kl_coeff == 0.02
    assert model.objective.pretraining_mix_coeff == 27.8

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    rejected_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids)
    assert tuple(logits.shape) == (2, 8, 64)

    reward_loss = model.reward_loss(chosen_input_ids=input_ids, rejected_input_ids=rejected_ids)
    assert reward_loss.ndim == 0

    advantages = torch.ones(2, 8, dtype=torch.float32)
    objective = model.objective(
        policy_logits=logits,
        reference_logits=model.reference_policy(input_ids),
        sampled_ids=input_ids,
        advantages=advantages,
        pretraining_logits=logits,
        pretraining_labels=input_ids,
    )
    assert objective.ndim == 0


def test_instructgpt_ppo_objective_uses_clipped_ratio_against_old_policy() -> None:
    from Llms.instructgpt import PPOPTXObjective

    objective = PPOPTXObjective(
        kl_coeff=0.0,
        pretraining_mix_coeff=0.0,
        clip_range=0.2,
    )
    policy_logits = torch.tensor([[[0.35, 0.0], [0.35, 0.0]]], dtype=torch.float32)
    old_policy_logits = torch.zeros_like(policy_logits)
    reference_logits = torch.tensor([[[-2.0, 0.0], [-2.0, 0.0]]], dtype=torch.float32)
    sampled_ids = torch.zeros((1, 2), dtype=torch.long)
    advantages = torch.ones((1, 2), dtype=torch.float32)

    loss = objective(
        policy_logits=policy_logits,
        old_policy_logits=old_policy_logits,
        reference_logits=reference_logits,
        sampled_ids=sampled_ids,
        advantages=advantages,
    )

    policy_logprobs = torch.log_softmax(policy_logits, dim=-1)[..., 0]
    old_logprobs = torch.log_softmax(old_policy_logits, dim=-1)[..., 0]
    ratios = torch.exp(policy_logprobs - old_logprobs)
    expected = -torch.minimum(ratios, torch.full_like(ratios, 1.2)).mean()

    assert objective.clip_range == pytest.approx(0.2)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6, abs=1e-6)


def test_instructgpt_initializes_value_model_from_reward_model() -> None:
    from Llms.instructgpt import InstructGPTConfig, InstructGPTModel

    model = InstructGPTModel(
        InstructGPTConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )

    reward_params = dict(model.reward_model.named_parameters())
    value_params = dict(model.value_model.named_parameters())

    assert reward_params.keys() == value_params.keys()
    for name in reward_params:
        assert torch.allclose(value_params[name], reward_params[name], atol=1e-6, rtol=1e-6)
        assert value_params[name].data_ptr() != reward_params[name].data_ptr()


def test_instructgpt_reward_model_can_zero_center_demo_rewards() -> None:
    from Llms.instructgpt import InstructGPTConfig, InstructGPTRewardModel

    model = InstructGPTRewardModel(
        InstructGPTConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    demonstrations = torch.randint(0, 64, (4, 6), dtype=torch.long)

    with torch.no_grad():
        model.set_reward_bias_from_demonstrations(demonstrations)
        rewards = model(demonstrations)

    assert model.reward_bias.shape == (1,)
    assert rewards.mean().item() == pytest.approx(0.0, abs=1e-5)


def test_instructgpt_reward_loss_supports_ranked_completion_batches() -> None:
    from Llms.instructgpt import InstructGPTConfig, InstructGPTModel

    model = InstructGPTModel(
        InstructGPTConfig(
            vocab_size=64,
            max_seq_len=6,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    completion_input_ids = torch.randint(0, 64, (2, 4, 6), dtype=torch.long)
    rankings = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

    loss = model.reward_loss_from_rankings(
        completion_input_ids=completion_input_ids,
        rankings=rankings,
    )

    with torch.no_grad():
        rewards = model.reward_model(completion_input_ids.reshape(-1, 6)).view(2, 4)
    expected_terms = []
    for batch_idx in range(rankings.shape[0]):
        for i in range(rankings.shape[1]):
            for j in range(i + 1, rankings.shape[1]):
                better = i if rankings[batch_idx, i] < rankings[batch_idx, j] else j
                worse = j if better == i else i
                expected_terms.append(
                    -torch.nn.functional.logsigmoid(
                        rewards[batch_idx, better] - rewards[batch_idx, worse]
                    )
                )
    expected = torch.stack(expected_terms).mean()

    assert loss.item() == pytest.approx(expected.item(), rel=1e-6, abs=1e-6)


def test_self_instruct_bootstraps_filters_and_wraps_instruction_tuning() -> None:
    from Llms.self_instruct import (
        SelfInstructConfig,
        SelfInstructDatasetBuilder,
        SelfInstructModel,
        format_self_instruct_prompt,
    )

    prompt = format_self_instruct_prompt("Write a haiku.", "about spring rain")
    assert "Write a haiku." in prompt
    assert "about spring rain" in prompt

    builder = SelfInstructDatasetBuilder(seed_instructions=["Summarize the passage."])
    assert builder.infer_task_type("Classify the sentiment of the review.") == "classification"
    assert builder.infer_task_type("Write a short poem about the moon.") == "generation"
    filtered = builder.filter_candidate_instructions(
        [
            "Summarize the passage.",
            "Translate the sentence to French.",
            "Translate the sentence to French.",
            "  ",
        ]
    )
    assert filtered == ["Translate the sentence to French."]

    example = builder.build_example(
        instruction="Classify the sentiment.",
        instance_input="The movie was fantastic.",
        output="positive",
        task_type="classification",
    )
    assert example.task_type == "classification"
    assert "Classify the sentiment." in example.prompt
    assert "The movie was fantastic." in example.prompt

    model = SelfInstructModel(
        SelfInstructConfig(
            vocab_size=64,
            max_seq_len=8,
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=64,
            dropout=0.0,
        )
    )
    assert model.base_model.__class__.__name__ == "FlanT5Model"

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    decoder_input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    assert tuple(logits.shape) == (2, 8, 64)


def test_self_instruct_uses_input_first_and_output_first_instance_generation_prompts() -> None:
    from Llms.self_instruct import SelfInstructDatasetBuilder

    builder = SelfInstructDatasetBuilder(seed_instructions=["Summarize the passage."])
    generation_prompt = builder.build_instance_generation_prompt(
        "Write a short poem about the moon.",
    )
    classification_prompt = builder.build_instance_generation_prompt(
        "Classify the sentiment of the review.",
        class_labels=("positive", "negative"),
    )

    assert generation_prompt.task_type == "generation"
    assert generation_prompt.approach == "input_first"
    assert generation_prompt.prompt.index("Input:") < generation_prompt.prompt.index("Output:")
    assert classification_prompt.task_type == "classification"
    assert classification_prompt.approach == "output_first"
    assert "Class label:" in classification_prompt.prompt
    assert "positive" in classification_prompt.prompt
    assert classification_prompt.prompt.index("Class label:") < classification_prompt.prompt.index("Input:")


def test_self_instruct_samples_bootstrap_batch_and_filters_modal_tasks() -> None:
    from Llms.self_instruct import SelfInstructDatasetBuilder

    builder = SelfInstructDatasetBuilder(
        seed_instructions=[
            "Summarize the passage.",
            "Translate the sentence to French.",
            "Write a haiku about spring.",
            "Explain gravity simply.",
            "Classify the sentiment of the review.",
            "List three causes of rain.",
            "Rewrite the paragraph in plain English.",
        ]
    )
    batch = builder.sample_bootstrap_batch(
        machine_generated_instructions=[
            "Describe the lifecycle of a butterfly.",
            "Generate a caption for an image of a dog.",
            "Write a product tagline.",
        ],
        seed_count=6,
        generated_count=2,
    )
    filtered = builder.filter_candidate_instructions(
        [
            "Generate a caption for an image of a dog.",
            "Summarize the passage.",
            "Write a product tagline.",
            "Create a video storyboard for a game trailer.",
        ]
    )

    assert len(batch) == 8
    assert batch[:6] == (
        "Summarize the passage.",
        "Translate the sentence to French.",
        "Write a haiku about spring.",
        "Explain gravity simply.",
        "Classify the sentiment of the review.",
        "List three causes of rain.",
    )
    assert batch[6:] == (
        "Describe the lifecycle of a butterfly.",
        "Write a product tagline.",
    )
    assert filtered == ["Write a product tagline."]


def test_self_instruct_uses_rouge_l_style_similarity_threshold_of_point_seven() -> None:
    from Llms.self_instruct import SelfInstructDatasetBuilder

    builder = SelfInstructDatasetBuilder(seed_instructions=["Summarize the passage."])
    similar = builder.rouge_l_similarity(
        "Summarize the passage briefly.",
        "Summarize the passage.",
    )
    dissimilar = builder.rouge_l_similarity(
        "Translate the sentence to French.",
        "Summarize the passage.",
    )

    assert builder.similarity_threshold == pytest.approx(0.7)
    assert similar > builder.similarity_threshold
    assert dissimilar < builder.similarity_threshold


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


def test_anthropic_models_helpful_harmless_preferences_and_online_rlhf() -> None:
    from Llms.anthropic import AnthropicConfig, AnthropicModel

    model = AnthropicModel(
        AnthropicConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
            helpfulness_mix=0.6,
            harmlessness_mix=0.4,
        )
    )

    assert model.objective_names == ("helpfulness", "harmlessness")
    assert model.supports_online_rlhf is True
    assert model.preference_model.helpfulness_head.out_features == 1
    assert model.preference_model.harmlessness_head.out_features == 1
    assert all(not p.requires_grad for p in model.reference_policy.parameters())

    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    rejected_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids)
    assert tuple(logits.shape) == (2, 8, 64)

    preference_scores = model.preference_model(input_ids)
    assert set(preference_scores) == {"helpfulness", "harmlessness"}
    combined_loss = model.preference_loss(
        chosen_input_ids=input_ids,
        rejected_input_ids=rejected_ids,
    )
    assert combined_loss.ndim == 0


def test_anthropic_preference_model_ignores_masked_tail_tokens() -> None:
    from Llms.anthropic import AnthropicConfig, AnthropicModel

    model = AnthropicModel(
        AnthropicConfig(
            vocab_size=64,
            max_seq_len=5,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)
    prefix = torch.tensor([[4, 5, 6, 0, 0]], dtype=torch.long)
    changed_tail = torch.tensor([[4, 5, 6, 31, 42]], dtype=torch.long)

    with torch.no_grad():
        prefix_scores = model.preference_model(prefix, attention_mask)
        changed_scores = model.preference_model(changed_tail, attention_mask)

    assert torch.allclose(
        prefix_scores["helpfulness"], changed_scores["helpfulness"], atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        prefix_scores["harmlessness"], changed_scores["harmlessness"], atol=1e-6, rtol=1e-6
    )


def test_anthropic_feedback_dataset_filters_weakest_labels_and_inverts_red_team_choice() -> None:
    from Llms.anthropic import AnthropicComparison, AnthropicFeedbackDataset

    dataset = AnthropicFeedbackDataset(
        comparisons=(
            AnthropicComparison(
                prompt="How do I draft a polite email?",
                response_a="Start with a greeting and be specific.",
                response_b="No idea.",
                task="helpfulness",
                selected="a",
                preference_strength="strong",
                source="base",
            ),
            AnthropicComparison(
                prompt="How do I make a dangerous weapon?",
                response_a="I can't help with building weapons.",
                response_b="Here is a parts list and assembly guide.",
                task="harmlessness",
                selected="b",
                preference_strength="medium",
                source="online",
            ),
            AnthropicComparison(
                prompt="Weak preference example",
                response_a="Option A",
                response_b="Option B",
                task="helpfulness",
                selected="b",
                preference_strength="weakest",
                source="rs",
            ),
        )
    )

    pairs = dataset.training_pairs()

    assert dataset.rejection_sampling_k == 16
    assert dataset.online_update_interval_days == 7
    assert pairs == (
        ("Start with a greeting and be specific.", "No idea.", "helpfulness"),
        ("I can't help with building weapons.", "Here is a parts list and assembly guide.", "harmlessness"),
    )
    assert dataset.source_counts() == {"base": 1, "online": 1, "rs": 1}


def test_anthropic_rejection_sampling_selects_best_candidate_by_objective() -> None:
    from Llms.anthropic import AnthropicConfig, AnthropicModel

    model = AnthropicModel(
        AnthropicConfig(
            vocab_size=64,
            max_seq_len=6,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
            helpfulness_mix=0.7,
            harmlessness_mix=0.3,
        )
    )
    candidate_input_ids = torch.randint(0, 64, (4, 6), dtype=torch.long)
    candidate_attention_mask = torch.ones_like(candidate_input_ids, dtype=torch.float32)

    with torch.no_grad():
        scores = model.preference_model(candidate_input_ids, candidate_attention_mask)
    expected_hh = (
        0.7 * scores["helpfulness"] + 0.3 * scores["harmlessness"]
    ).argmax().item()
    expected_helpfulness = scores["helpfulness"].argmax().item()
    expected_harmlessness = scores["harmlessness"].argmax().item()

    hh_choice = model.rejection_sample(
        candidate_input_ids=candidate_input_ids,
        candidate_attention_mask=candidate_attention_mask,
        objective="hh",
    )
    helpful_choice = model.rejection_sample(
        candidate_input_ids=candidate_input_ids,
        candidate_attention_mask=candidate_attention_mask,
        objective="helpfulness",
    )
    harmless_choice = model.rejection_sample(
        candidate_input_ids=candidate_input_ids,
        candidate_attention_mask=candidate_attention_mask,
        objective="harmlessness",
    )

    assert hh_choice.selected_index == expected_hh
    assert helpful_choice.selected_index == expected_helpfulness
    assert harmless_choice.selected_index == expected_harmlessness
    assert hh_choice.num_candidates == 4


def test_imagen_uses_frozen_text_encoder_cascade_and_dynamic_thresholding() -> None:
    from Llms.imagen import ImagenConfig, ImagenModel, dynamic_threshold

    values = torch.tensor([[-2.0, -0.5, 0.5, 3.0]], dtype=torch.float32)
    clipped = dynamic_threshold(values, percentile=0.75)
    assert clipped.abs().max() <= 1.0 + 1e-6

    model = ImagenModel(
        ImagenConfig(
            text_vocab_size=64,
            text_hidden_size=32,
            base_channels=16,
            image_size=16,
            superres_sizes=(32, 64),
            cond_drop_prob=0.1,
        )
    )
    assert all(not p.requires_grad for p in model.text_encoder.parameters())
    assert len(model.super_resolution_models) == 2
    assert model.base_diffusion.unet.cross_attention is not None
    assert model.base_diffusion.uses_classifier_free_guidance is True
    assert model.uses_dynamic_thresholding is True

    token_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    images = model(token_ids)
    assert tuple(images.shape) == (2, 3, 64, 64)


def test_imagen_condition_dropout_can_remove_text_conditioning() -> None:
    from Llms.imagen import ImagenConfig, ImagenModel

    model = ImagenModel(
        ImagenConfig(
            text_vocab_size=64,
            text_hidden_size=32,
            base_channels=16,
            image_size=16,
            superres_sizes=(32, 64),
            cond_drop_prob=1.0,
        )
    )
    model.train()
    token_ids_a = torch.zeros((1, 8), dtype=torch.long)
    token_ids_b = torch.full((1, 8), 7, dtype=torch.long)

    with torch.no_grad():
        image_a = model(token_ids_a)
        image_b = model(token_ids_b)

    assert torch.allclose(image_a, image_b, atol=1e-6, rtol=1e-6)


def test_vilt_uses_patch_projection_single_stream_and_modality_embeddings() -> None:
    from Llms.vilt import ViLTConfig, ViLTModel

    model = ViLTModel(
        ViLTConfig(
            vocab_size=64,
            image_size=32,
            patch_size=16,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )

    assert model.uses_convolution is False
    assert model.uses_region_supervision is False
    assert model.modal_type_embeddings.weight.shape == (2, 32)
    assert model.patch_projection.kernel_size == (16, 16)
    assert model.transformer.layers[0].self_attn.num_heads == 4
    assert model.word_patch_alignment_head is not None

    image = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 64, (2, 6), dtype=torch.long)
    outputs = model(image=image, input_ids=input_ids)
    assert {"cls_embedding", "text_embeddings", "image_embeddings"}.issubset(outputs)
    assert tuple(outputs["cls_embedding"].shape) == (2, 32)


def test_vilt_exposes_word_patch_alignment_scores_over_patches() -> None:
    from Llms.vilt import ViLTConfig, ViLTModel

    model = ViLTModel(
        ViLTConfig(
            vocab_size=64,
            image_size=32,
            patch_size=16,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    image = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, 64, (2, 6), dtype=torch.long)

    outputs = model(image=image, input_ids=input_ids)

    assert model.word_patch_alignment_head.out_features == model.num_patches
    assert "word_patch_alignment" in outputs
    assert tuple(outputs["word_patch_alignment"].shape) == (2, 6, 4)


def test_chinchilla_recommends_more_tokens_for_same_compute_budget() -> None:
    from Llms.chinchilla import ChinchillaConfig, ChinchillaPlanner

    planner = ChinchillaPlanner(ChinchillaConfig())
    canonical = planner.plan_for_parameters(70_000_000_000)
    gopher_budget = planner.training_flops(
        parameters=280_000_000_000,
        tokens=300_000_000_000,
    )
    optimal = planner.plan_for_compute(gopher_budget)

    assert canonical.tokens == 1_400_000_000_000
    assert canonical.tokens_per_parameter == pytest.approx(20.0)
    assert optimal.parameters < 280_000_000_000
    assert optimal.tokens > 300_000_000_000
    assert optimal.compute_budget_flops == pytest.approx(gopher_budget)


def test_gpipe_balances_cells_builds_pipeline_schedule_and_matches_sequential() -> None:
    from Llms.gpipe import GPipeConfig, GPipeSequential

    layers = torch.nn.ModuleList(
        [
            torch.nn.Linear(4, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        ]
    )
    reference = torch.nn.Sequential(*layers)
    pipeline = GPipeSequential(
        layers,
        GPipeConfig(num_partitions=3, micro_batches=4, rematerialization=True),
    )
    x = torch.randn(8, 4)
    schedule = pipeline.pipeline_schedule()

    with torch.no_grad():
        expected = reference(x)
        actual = pipeline(x)

    assert pipeline.partition_sizes == [2, 2, 1]
    assert pipeline.rematerialization is True
    assert pipeline.bubble_steps == 2
    assert len(schedule) == 6
    assert schedule[0] == ((0, 0),)
    assert schedule[-1] == ((2, 3),)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_pathways_maps_virtual_devices_gang_schedules_and_traces_fused_programs() -> None:
    from Llms.pathways import PathwaysConfig, PathwaysProgram, PathwaysRuntime, PathwaysTracer, VirtualDevice

    devices = [
        VirtualDevice(logical_id="v2", island="island-b", physical_device="pod-b:7"),
        VirtualDevice(logical_id="v0", island="island-a", physical_device="pod-a:3"),
        VirtualDevice(logical_id="v1", island="island-b", physical_device="pod-b:1"),
        VirtualDevice(logical_id="v3", island="island-a", physical_device="pod-a:8"),
    ]
    runtime = PathwaysRuntime(PathwaysConfig(interleave_quantum=1), devices)
    tracer = PathwaysTracer()
    tracer.add_compiled("embed")
    tracer.add_compiled("dispatch")
    tracer.add_compiled("decode")
    fused = tracer.fuse("serve", required_devices=2)
    retrieval = PathwaysProgram(name="retrieve", stages=("lookup", "merge"), required_devices=2)

    mapped = runtime.map_virtual_devices(fused)
    gang = runtime.gang_schedule([fused, retrieval])
    interleaved = runtime.interleave([fused, retrieval])

    assert [device.logical_id for device in mapped] == ["v0", "v1"]
    assert fused.compiled_functions == ("embed", "dispatch", "decode")
    assert fused.stages == ("embed", "dispatch", "decode")
    assert gang[0].program.name == "serve"
    assert len(gang[0].devices) == 2
    assert gang[0].islands == ("island-a", "island-b")
    assert [step.program_name for step in interleaved[:4]] == ["serve", "retrieve", "serve", "retrieve"]


def test_zero_partitions_states_by_stage_and_reconstructs_parameters() -> None:
    from Llms.zero import ZeROConfig, ZeROEngine

    parameter = torch.arange(12, dtype=torch.float32)
    gradient = parameter + 0.5
    optimizer_state = {
        "momentum": parameter + 1.0,
        "variance": parameter + 2.0,
    }

    stage1 = ZeROEngine(ZeROConfig(stage=1, world_size=3, rank=1))
    shard1 = stage1.partition_states(
        parameter=parameter,
        gradient=gradient,
        optimizer_state=optimizer_state,
    )
    assert shard1.plan.partitions_optimizer_state is True
    assert shard1.plan.partitions_gradients is False
    assert shard1.plan.partitions_parameters is False
    assert shard1.parameter.numel() == 12
    assert shard1.gradient.numel() == 12
    assert all(value.numel() == 4 for value in shard1.optimizer_state.values())

    stage2 = ZeROEngine(ZeROConfig(stage=2, world_size=3, rank=1))
    shard2 = stage2.partition_states(
        parameter=parameter,
        gradient=gradient,
        optimizer_state=optimizer_state,
    )
    assert shard2.gradient.numel() == 4
    assert shard2.parameter.numel() == 12

    stage3_shards = [
        ZeROEngine(ZeROConfig(stage=3, world_size=3, rank=rank)).partition_states(
            parameter=parameter,
            gradient=gradient,
            optimizer_state=optimizer_state,
        )
        for rank in range(3)
    ]
    assert all(shard.parameter.numel() == 4 for shard in stage3_shards)
    reconstructed = stage3_shards[0].engine.gather_parameters(
        [shard.parameter for shard in stage3_shards]
    )
    assert torch.allclose(reconstructed, parameter, atol=1e-6, rtol=1e-6)


def test_parameter_server_supports_sparse_pull_push_and_ssp_progress() -> None:
    from Llms.parameter_server import ParameterServer, ParameterServerConfig

    server = ParameterServer(
        {"embedding": torch.tensor([1.0, 2.0, 3.0, 4.0])},
        ParameterServerConfig(num_workers=2, consistency="ssp", staleness=1),
    )
    worker0 = server.register_worker("worker0")
    worker1 = server.register_worker("worker1")

    pulled = worker0.pull("embedding", indices=torch.tensor([1, 3], dtype=torch.long))
    worker0.push(
        "embedding",
        indices=torch.tensor([1, 3], dtype=torch.long),
        values=torch.tensor([0.5, -1.0]),
    )

    assert torch.allclose(pulled, torch.tensor([2.0, 4.0]), atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        server.parameters["embedding"],
        torch.tensor([1.0, 2.5, 3.0, 3.0]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert worker0.finish_step() is True
    assert worker0.finish_step() is True
    assert worker0.finish_step() is False
    assert worker1.finish_step() is True
    assert worker0.finish_step() is True


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


def test_helm_tracks_multi_metric_coverage_and_preserves_prompts_and_completions() -> None:
    from Llms.helm import HELMEvaluator, HELMRun, HELMScenario

    evaluator = HELMEvaluator()
    run_qa = HELMRun(
        model_name="demo-lm",
        scenario=HELMScenario(name="qa", kind="core"),
        metric_scores={
            "accuracy": 0.8,
            "calibration": 0.7,
            "robustness": 0.6,
            "fairness": 0.9,
            "bias": 0.2,
            "toxicity": 0.1,
            "efficiency": 0.95,
        },
        prompts=("Q: 2+2?",),
        completions=("A: 4",),
    )
    run_reasoning = HELMRun(
        model_name="demo-lm",
        scenario=HELMScenario(name="reasoning", kind="targeted"),
        metric_scores={
            "accuracy": 0.7,
            "calibration": 0.65,
            "robustness": 0.6,
            "fairness": 0.85,
            "bias": 0.25,
            "toxicity": 0.15,
        },
        prompts=("Solve step by step.",),
        completions=("Let's think through it.",),
    )
    report = evaluator.evaluate([run_qa, run_reasoning])

    assert evaluator.metric_categories == (
        "accuracy",
        "calibration",
        "robustness",
        "fairness",
        "bias",
        "toxicity",
        "efficiency",
    )
    assert report.coverage == pytest.approx(13 / 14)
    assert report.metric_matrix["qa"]["accuracy"] == pytest.approx(0.8)
    assert report.metric_matrix["reasoning"]["efficiency"] is None
    assert report.prompt_logs["qa"] == ("Q: 2+2?",)
    assert report.completion_logs["reasoning"] == ("Let's think through it.",)


def test_helm_aggregates_model_leaderboard_kind_coverage_and_per_model_matrices() -> None:
    from Llms.helm import HELMEvaluator, HELMRun, HELMScenario

    evaluator = HELMEvaluator()
    report = evaluator.evaluate(
        [
            HELMRun(
                model_name="model-a",
                scenario=HELMScenario(
                    name="qa",
                    kind="core",
                    metrics=("accuracy", "calibration"),
                ),
                metric_scores={"accuracy": 0.8, "calibration": 0.6},
            ),
            HELMRun(
                model_name="model-a",
                scenario=HELMScenario(
                    name="reasoning",
                    kind="targeted",
                    metrics=("accuracy", "robustness"),
                ),
                metric_scores={"accuracy": 0.7, "robustness": 0.5},
            ),
            HELMRun(
                model_name="model-b",
                scenario=HELMScenario(
                    name="qa",
                    kind="core",
                    metrics=("accuracy", "calibration"),
                ),
                metric_scores={"accuracy": 0.55},
            ),
        ]
    )

    assert report.model_metric_matrix["model-a"]["reasoning"]["robustness"] == pytest.approx(0.5)
    assert report.model_metric_matrix["model-b"]["qa"]["calibration"] is None
    assert report.scenario_kind_coverage["core"].coverage == pytest.approx(3 / 4)
    assert report.scenario_kind_coverage["targeted"].coverage == pytest.approx(1.0)
    assert report.leaderboard[0].model_name == "model-a"
    assert report.leaderboard[0].coverage == pytest.approx(1.0)
    assert report.leaderboard[0].metric_averages["accuracy"] == pytest.approx(0.75)
    assert report.leaderboard[0].macro_average == pytest.approx((0.75 + 0.6 + 0.5) / 3)
    assert report.leaderboard[1].model_name == "model-b"
    assert report.leaderboard[1].coverage == pytest.approx(0.5)


def test_pile_exposes_22_components_and_normalized_mixture_shares() -> None:
    from Llms.pile import PileConfig, PileMixture, canonical_pile_components

    components = canonical_pile_components()
    mixture = PileMixture(PileConfig(), components)
    shares = mixture.normalized_shares()
    counts = mixture.allocate(1000)

    assert len(components) == 22
    assert "Pile-CC" in shares
    assert "ArXiv" in shares
    assert "Github" in shares
    assert "Stack Exchange" in shares
    assert "Enron Emails" in shares
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["Pile-CC"] > shares["Enron Emails"]
    assert sum(counts.values()) == 1000
    assert counts["Pile-CC"] > counts["Enron Emails"]


def test_the_stack_filters_permissive_code_deduplicates_and_supports_opt_out() -> None:
    from Llms.the_stack import StackFile, TheStackConfig, TheStackDataset

    dataset = TheStackDataset(
        files=[
            StackFile(
                repo_name="alpha",
                path="alpha/main.py",
                language="Python",
                license="MIT",
                content="def add(a, b):\n    return a + b\n",
            ),
            StackFile(
                repo_name="beta",
                path="beta/main.py",
                language="Python",
                license="Apache-2.0",
                content="def add(a,b): return a+b\n",
            ),
            StackFile(
                repo_name="gamma",
                path="gamma/app.ts",
                language="TypeScript",
                license="GPL-3.0",
                content="export const x = 1;\n",
            ),
        ],
        config=TheStackConfig(near_dedup_threshold=0.85),
    )

    permissive = dataset.permissive_subset()
    deduped = permissive.near_deduplicate()
    opted_out = deduped.remove_repositories({"alpha"})
    language_bytes = deduped.language_bytes()

    assert len(permissive.files) == 2
    assert len(deduped.files) == 1
    assert deduped.files[0].license in {"MIT", "Apache-2.0"}
    assert len(opted_out.files) == 0
    assert language_bytes["Python"] > 0


def test_scienceqa_builds_cot_prompt_and_scores_multimodal_choices() -> None:
    from Llms.scienceqa import ScienceQAConfig, ScienceQAExample, ScienceQAModel, format_scienceqa_prompt

    example = ScienceQAExample(
        question="Which planet is known as the Red Planet?",
        choices=("Earth", "Mars", "Jupiter", "Venus"),
        answer_index=1,
        lecture="Mars appears reddish because of iron oxide on its surface.",
        explanation="Mars is called the Red Planet because its surface contains iron oxide.",
        text_context="The image shows a reddish rocky planet.",
        has_image=True,
    )
    prompt = format_scienceqa_prompt(example)
    model = ScienceQAModel(
        ScienceQAConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            image_feature_dim=12,
            num_choices=4,
        )
    )
    question_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    image_features = torch.randn(2, 12)
    choice_logits = model(question_ids=question_ids, image_features=image_features)

    assert "Let's think step by step." in prompt
    assert "Lecture:" in prompt
    assert "Choices:" in prompt
    assert example.has_image is True
    assert tuple(choice_logits.shape) == (2, 4)


def test_dolly_uses_databricks_prompt_sections_and_pythia_backbone() -> None:
    from Llms.dolly import DollyConfig, DollyModel, format_dolly_prompt

    prompt = format_dolly_prompt(
        instruction="Summarize the paragraph.",
        context="Transformers process tokens in parallel.",
    )
    model = DollyModel(
        DollyConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids)

    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert "### Response:" in prompt
    assert model.dataset_name == "databricks-dolly-15k"
    assert model.human_generated_data is True
    assert model.base_model.__class__.__name__ == "PythiaModel"
    assert tuple(logits.shape) == (2, 8, 64)


def test_gpt4all_exposes_distillation_curation_and_quantized_lora_metadata() -> None:
    from Llms.gpt4all import GPT4AllConfig, GPT4AllModel, format_gpt4all_prompt

    prompt = format_gpt4all_prompt("Write a haiku about debugging.")
    model = GPT4AllModel(
        GPT4AllConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
            lora_rank=8,
            quantization_bits=4,
        )
    )
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids)

    assert "### User:" in prompt
    assert "### Assistant:" in prompt
    assert model.data_curation.initial_pairs == 1_000_000
    assert model.data_curation.cleaned_pairs == 806_199
    assert model.data_curation.final_pairs == 437_605
    assert model.data_curation.distillation_source == "gpt-3.5-turbo"
    assert model.training_strategy == "lora"
    assert model.quantization_bits == 4
    assert model.base_model.__class__.__name__ == "LLaMAModel"
    assert tuple(logits.shape) == (2, 8, 64)


def test_chain_of_thought_formats_rationales_and_votes_with_self_consistency() -> None:
    from Llms.chain_of_thought import (
        ChainOfThoughtConfig,
        ChainOfThoughtExample,
        ChainOfThoughtReasoner,
        SelfConsistencyDecoder,
        extract_final_answer,
        format_chain_of_thought_prompt,
    )

    demonstrations = (
        ChainOfThoughtExample(
            question="Roger has 3 apples and buys 2 more. How many apples does he have?",
            rationale="Roger starts with 3 and adds 2, so 3 + 2 = 5.",
            answer="5",
        ),
    )
    prompt = format_chain_of_thought_prompt(
        question="A box has 4 red balls and 3 blue balls. How many balls are there?",
        demonstrations=demonstrations,
    )
    traces = (
        "Let's think step by step. 4 + 3 = 7. Therefore, the answer is 7.",
        "Reasoning: there are seven balls total. Answer: 7",
        "Let's think step by step. I incorrectly guessed 6. Answer: 6",
    )
    decoder = SelfConsistencyDecoder(temperature=0.4, num_samples=3)
    reasoner = ChainOfThoughtReasoner(ChainOfThoughtConfig(temperature=0.4, num_samples=3))

    assert "Let's think step by step." in prompt
    assert "Question:" in prompt
    assert "Reasoning:" in prompt
    assert extract_final_answer(traces[0]) == "7"
    assert decoder.majority_vote(traces) == "7"
    assert reasoner.aggregate_answers(traces) == "7"


def test_gpt4all_j_uses_gpt_j_metadata_and_creative_augmentation() -> None:
    from Llms.gpt4all_j import GPT4AllJConfig, GPT4AllJModel, build_creative_prompt, format_gpt4all_j_prompt

    creative_prompt = build_creative_prompt(
        genre="poem",
        topic="debugging",
        style="Sappho",
    )
    prompt = format_gpt4all_j_prompt("Write a short poem about debugging.")
    model = GPT4AllJModel(
        GPT4AllJConfig(
            vocab_size=64,
            max_seq_len=8,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
            quantization_bits=4,
        )
    )
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids)

    assert "Write a poem about debugging in the style of Sappho." == creative_prompt
    assert "### Prompt:" in prompt
    assert "### Response:" in prompt
    assert model.data_curation.base_checkpoint == "gpt-j-6.7b"
    assert model.data_curation.license == "Apache-2.0"
    assert "poems" in model.data_curation.creative_domains
    assert model.data_curation.dataset_points == 800_000
    assert model.quantization_bits == 4
    assert model.base_model.__class__.__name__ == "GPTJBackbone"
    assert tuple(logits.shape) == (2, 8, 64)


def test_mtf_materializes_prompt_templates_and_tracks_zero_shot_tasks() -> None:
    from Llms.mtf import MTFConfig, MTFMixture, MTFModel, MTFTask, PromptTemplate

    template = PromptTemplate(
        name="nli-basic",
        input_template="If {premise} is true, is it also true that {hypothesis}?",
        target_template="{answer}",
        metadata={"choices": ("yes", "no")},
    )
    rendered = template.materialize(
        {
            "premise": "All whales are mammals.",
            "hypothesis": "Whales are mammals.",
            "answer": "yes",
        }
    )
    train_task = MTFTask(name="nli", templates=(template,))
    held_out_task = MTFTask(name="summarization", templates=(template,), held_out=True)
    mixture = MTFMixture(train_tasks=(train_task,), evaluation_tasks=(held_out_task,))
    model = MTFModel(
        MTFConfig(
            vocab_size=64,
            max_seq_len=8,
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=64,
            dropout=0.0,
        )
    )
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    decoder_input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    assert rendered.input_text.startswith("If All whales are mammals.")
    assert rendered.target_text == "yes"
    assert mixture.seen_task_names() == ("nli",)
    assert mixture.zero_shot_task_names() == ("summarization",)
    assert model.training_objective == "multitask_prompted_training"
    assert model.base_model.__class__.__name__ == "T5Model"
    assert tuple(logits.shape) == (2, 8, 64)


def test_segment_anything_supports_promptable_masks_and_sa1b_metadata() -> None:
    from Llms.segment_anything import (
        SAMAutomaticMaskGenerator,
        SAMConfig,
        SAMPrompt,
        SegmentAnythingModel,
    )

    model = SegmentAnythingModel(
        SAMConfig(
            image_size=8,
            patch_size=4,
            embed_dim=32,
            num_heads=4,
            num_prompt_masks=3,
            iou_head_hidden_dim=16,
        )
    )
    prompts = (
        SAMPrompt(point=(2, 2)),
        SAMPrompt(box=(1, 1, 6, 6)),
    )
    image = torch.randn(2, 3, 8, 8)
    output = model(image=image, prompts=prompts)
    generator = SAMAutomaticMaskGenerator(points_per_side=4)
    grid_prompts = generator.build_grid_prompts(image_size=8)

    assert model.data_engine.num_images == 11_000_000
    assert model.data_engine.num_masks == 1_100_000_000
    assert model.prompt_encoder.supports_text_prompts is True
    assert tuple(output["mask_logits"].shape) == (2, 3, 8, 8)
    assert tuple(output["iou_scores"].shape) == (2, 3)
    assert len(grid_prompts) == 16
    assert all(prompt.point is not None for prompt in grid_prompts)


def test_blip_unifies_itc_itm_lm_and_capfilt_curation() -> None:
    from Llms.blip import BLIPConfig, BLIPModel, CapFiltPair, CapFiltPipeline

    pipeline = CapFiltPipeline(score_threshold=0.7)
    pairs = (
        CapFiltPair(image_id="img-1", caption="a cat on a sofa", score=0.92),
        CapFiltPair(image_id="img-2", caption="blurry noise", score=0.35),
    )
    filtered = pipeline.filter_pairs(pairs)
    model = BLIPModel(
        BLIPConfig(
            vocab_size=64,
            max_seq_len=8,
            image_feat_dim=12,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
        )
    )
    image_features = torch.randn(2, 4, 12)
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    output = model(image_features=image_features, input_ids=input_ids)

    assert len(filtered) == 1
    assert filtered[0].image_id == "img-1"
    assert model.training_tasks == ("itc", "itm", "lm")
    assert model.image_encoder is not None
    assert model.multimodal_encoder is not None
    assert tuple(output["image_text_contrastive"].shape) == (2, 32)
    assert tuple(output["image_text_matching"].shape) == (2, 2)
    assert tuple(output["logits"].shape) == (2, 8, 64)


def test_instructblip_conditions_queries_on_instruction_tokens() -> None:
    from Llms.instructblip import InstructBLIPConfig, InstructBLIPModel, format_instructblip_prompt

    prompt = format_instructblip_prompt(
        instruction="Answer in one sentence.",
        question="What is happening in the image?",
    )
    config = InstructBLIPConfig(
        vocab_size=64,
        max_seq_len=8,
        llm_dim=32,
        num_heads=4,
        num_layers=2,
        intermediate_size=64,
        image_feat_dim=12,
        num_query_tokens=6,
        qformer_hidden_size=24,
        dropout=0.0,
    )
    model = InstructBLIPModel(config)
    image_features = torch.randn(2, 4, 12)
    instruction_ids = torch.randint(0, 64, (2, 5), dtype=torch.long)
    input_ids = torch.randint(0, 64, (2, 8), dtype=torch.long)
    visual_queries = model.encode_image_with_instruction(
        image_features=image_features,
        instruction_ids=instruction_ids,
    )
    logits = model(
        image_features=image_features,
        instruction_ids=instruction_ids,
        input_ids=input_ids,
    )

    assert prompt.startswith("Instruction:")
    assert "Question:" in prompt
    assert model.instruction_tuning_enabled is True
    assert model.llm_frozen is True
    assert model.q_former.__class__.__name__ == "InstructionAwareQFormer"
    assert tuple(visual_queries.shape) == (2, 6, 24)
    assert tuple(logits.shape) == (2, 8, 64)


def test_redpajama_tracks_seven_data_slices_and_token_allocation() -> None:
    from Llms.red_pajama import RedPajamaConfig, RedPajamaDataset, canonical_red_pajama_slices

    slices = canonical_red_pajama_slices()
    dataset = RedPajamaDataset(
        RedPajamaConfig(total_tokens=1_200_000_000_000),
        slices,
    )
    allocation = dataset.allocate_tokens(1_000)
    slice_names = {data_slice.name for data_slice in slices}

    assert len(slices) == 7
    assert slice_names == {
        "CommonCrawl",
        "C4",
        "GitHub",
        "arXiv",
        "Books",
        "Wikipedia",
        "StackExchange",
    }
    assert dataset.license_filtered_sources() == ("GitHub",)
    assert sum(allocation.values()) == 1_000
    assert dataset.config.total_tokens == 1_200_000_000_000


def test_starcoder_formats_fill_in_middle_and_exposes_code_llm_metadata() -> None:
    from Llms.starcoder import StarCoderConfig, StarCoderModel, format_fim_prompt

    prompt = format_fim_prompt(
        prefix="def add(a, b):\n    ",
        suffix="\nprint(add(1, 2))",
    )
    model = StarCoderModel(
        StarCoderConfig(
            vocab_size=64,
            max_seq_len=16,
            hidden_size=32,
            num_heads=4,
            num_layers=2,
            intermediate_size=64,
            dropout=0.0,
        )
    )
    input_ids = torch.randint(0, 64, (2, 16), dtype=torch.long)
    logits = model(input_ids=input_ids)

    assert "<fim_prefix>" in prompt
    assert "<fim_suffix>" in prompt
    assert "<fim_middle>" in prompt
    assert model.supports_fill_in_the_middle is True
    assert model.data_config.context_window == 8192
    assert model.data_config.num_languages >= 80
    assert model.data_config.license == "OpenRAIL"
    assert "GitHub issues" in model.data_config.training_sources
    assert tuple(logits.shape) == (2, 16, 64)


def test_prompt_engineering_guide_builds_structured_and_few_shot_prompts() -> None:
    from Llms.prompt_engineering_guide import (
        PromptEngineeringConfig,
        PromptEngineeringGuide,
        PromptExample,
        PromptTemplate,
    )

    guide = PromptEngineeringGuide(PromptEngineeringConfig())
    template = PromptTemplate(
        instruction="Translate the text below to Spanish:",
        context='Text: "hello!"',
        output_indicator="Output:",
    )
    prompt = template.render(delimiter="###")
    few_shot = guide.build_few_shot_prompt(
        examples=(
            PromptExample(input_text="This is awesome!", output_text="Positive"),
            PromptExample(input_text="This is bad!", output_text="Negative"),
        ),
        query="What a horrible show!",
    )
    factual = guide.recommend_sampling(task_type="factual_qa")
    creative = guide.recommend_sampling(task_type="creative_writing")

    assert "### Instruction ###" in prompt
    assert "### Context ###" in prompt
    assert "### Output ###" in prompt
    assert "This is awesome! // Positive" in few_shot
    assert "What a horrible show!" in few_shot
    assert factual.temperature < creative.temperature
    assert factual.top_p <= creative.top_p


def test_openassistant_builds_conversation_trees_and_preference_pairs() -> None:
    from Llms.openassistant import (
        OpenAssistantConfig,
        OpenAssistantConversationTree,
        OpenAssistantDataset,
        OpenAssistantMessage,
        OpenAssistantPreference,
    )

    root = OpenAssistantMessage(
        message_id="root",
        role="user",
        text="Explain transformers.",
        children=(
            OpenAssistantMessage(message_id="a1", role="assistant", text="Transformers use self-attention."),
            OpenAssistantMessage(message_id="a2", role="assistant", text="They are a kind of recurrent network."),
        ),
    )
    tree = OpenAssistantConversationTree(root=root)
    dataset = OpenAssistantDataset(
        OpenAssistantConfig(
            languages=35,
            messages=161_443,
            quality_ratings=461_292,
            complete_trees=10_000,
        ),
        trees=(tree,),
        preferences=(
            OpenAssistantPreference(chosen_id="a1", rejected_id="a2", score_gap=0.7),
        ),
    )

    flattened = tree.flatten_messages()
    pairs = dataset.preference_pairs()

    assert [message.message_id for message in flattened] == ["root", "a1", "a2"]
    assert pairs == (("a1", "a2"),)
    assert dataset.config.languages == 35
    assert dataset.config.messages == 161_443
    assert dataset.config.quality_ratings == 461_292


def test_openassistant_resolves_paths_message_lookup_and_preference_examples() -> None:
    from Llms.openassistant import (
        OpenAssistantConfig,
        OpenAssistantConversationTree,
        OpenAssistantDataset,
        OpenAssistantMessage,
        OpenAssistantPreference,
    )

    tree = OpenAssistantConversationTree(
        root=OpenAssistantMessage(
            message_id="root",
            role="user",
            text="Explain transformers.",
            children=(
                OpenAssistantMessage(
                    message_id="plan",
                    role="assistant",
                    text="Transformers use attention blocks.",
                    children=(
                        OpenAssistantMessage(
                            message_id="follow",
                            role="user",
                            text="Use an analogy.",
                            children=(
                                OpenAssistantMessage(
                                    message_id="chosen",
                                    role="assistant",
                                    text="Like reading every word in a sentence together.",
                                ),
                                OpenAssistantMessage(
                                    message_id="rejected",
                                    role="assistant",
                                    text="They are recurrent networks.",
                                ),
                            ),
                        ),
                    ),
                ),
                OpenAssistantMessage(
                    message_id="bad",
                    role="assistant",
                    text="Transformers are a type of recurrent model.",
                ),
            ),
        )
    )
    dataset = OpenAssistantDataset(
        OpenAssistantConfig(),
        trees=(tree,),
        preferences=(
            OpenAssistantPreference(chosen_id="chosen", rejected_id="rejected", score_gap=0.8),
        ),
    )

    root_to_leaf = tree.root_to_leaf_paths()
    chosen_path = tree.path_to("chosen")
    example = dataset.preference_examples()[0]

    assert [[message.message_id for message in path] for path in root_to_leaf] == [
        ["root", "plan", "follow", "chosen"],
        ["root", "plan", "follow", "rejected"],
        ["root", "bad"],
    ]
    assert [message.message_id for message in chosen_path] == ["root", "plan", "follow", "chosen"]
    assert dataset.message_by_id("follow").text == "Use an analogy."
    assert [message.message_id for message in example.context_messages] == ["root", "plan", "follow"]
    assert example.chosen_message.text == "Like reading every word in a sentence together."
    assert example.rejected_message.text == "They are recurrent networks."


def test_llm_survey_organizes_major_aspects_benchmarks_and_resources() -> None:
    from Llms.llm_survey import LLMSurveyGuide

    guide = LLMSurveyGuide()
    aspect_names = tuple(aspect.name for aspect in guide.major_aspects())
    benchmark_names = tuple(benchmark.name for benchmark in guide.benchmarks())
    resource_kinds = tuple(resource.kind for resource in guide.resources())

    assert aspect_names == (
        "pre-training",
        "adaptation tuning",
        "utilization",
        "capacity evaluation",
    )
    assert benchmark_names == ("MMLU", "BIG-bench", "HELM")
    assert "checkpoints" in resource_kinds
    assert "corpora" in resource_kinds
    assert "tooling" in resource_kinds


def test_llm_timeline_filters_milestones_by_date_and_category() -> None:
    from Llms.llm_timeline import LLMTimeline, canonical_llm_timeline_entries

    timeline = LLMTimeline(canonical_llm_timeline_entries())
    march_2023 = timeline.filter(year=2023, month=3)
    labels = tuple(entry.label for entry in march_2023)
    categories = {entry.category for entry in march_2023}

    assert "Bard" in labels
    assert "OpenAssistant" in labels
    assert "GPT4All" in labels
    assert "StarCoderData" in labels
    assert categories <= {"model", "dataset", "survey", "tool"}


def test_bard_exposes_early_experiment_modes_and_principled_responses() -> None:
    from Llms.bard import BardConfig, BardSession, format_bard_response

    session = BardSession(BardConfig())
    reply = format_bard_response(
        mode="creativity",
        content="Here is a brainstorming outline for your story.",
    )

    assert session.config.experiment_stage == "early experiment"
    assert session.modes() == ("productivity", "creativity", "curiosity")
    assert "AI Principles" in session.safety_note()
    assert reply.startswith("[creativity]")
    assert "brainstorming outline" in reply


def test_ai_bubbles_registry_filters_open_closed_and_chinchilla_scale_models() -> None:
    from Llms.ai_bubbles import AIBubblesRegistry, canonical_ai_bubbles_entries

    registry = AIBubblesRegistry(canonical_ai_bubbles_entries())
    labels = {entry.label for entry in registry.entries}
    open_labels = {entry.label for entry in registry.open_models()}
    chinchilla_labels = {entry.label for entry in registry.chinchilla_scale_models()}

    assert {"Bard", "GPT-4", "LLaMA", "PaLM"} <= labels
    assert "LLaMA" in open_labels
    assert "GPT-4" not in open_labels
    assert "LLaMA" in chinchilla_labels


def test_oasst1_aliases_openassistant_conversations_dataset() -> None:
    from Llms.oasst1 import OASST1Config, OASST1Dataset
    from Llms.openassistant import OpenAssistantConversationTree, OpenAssistantMessage, OpenAssistantPreference

    tree = OpenAssistantConversationTree(
        root=OpenAssistantMessage(
            message_id="root",
            role="user",
            text="Hello",
            children=(
                OpenAssistantMessage(message_id="assistant-1", role="assistant", text="Hi there"),
                OpenAssistantMessage(
                    message_id="assistant-2",
                    role="assistant",
                    text="Hello there, nice to meet you.",
                ),
            ),
        )
    )
    dataset = OASST1Dataset(
        OASST1Config(),
        trees=(tree,),
        preferences=(OpenAssistantPreference(chosen_id="assistant-2", rejected_id="assistant-1", score_gap=0.5),),
    )
    example = dataset.preference_examples()[0]

    assert dataset.config.dataset_name == "OpenAssistant Conversations Dataset"
    assert dataset.preference_pairs() == (("assistant-2", "assistant-1"),)
    assert dataset.flattened_messages()[0].message_id == "root"
    assert dataset.message_by_id("assistant-2").text == "Hello there, nice to meet you."
    assert [message.message_id for message in example.context_messages] == ["root"]
