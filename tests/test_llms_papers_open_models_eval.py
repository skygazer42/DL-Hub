import pytest

torch = pytest.importorskip("torch")


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
