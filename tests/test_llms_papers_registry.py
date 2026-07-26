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
