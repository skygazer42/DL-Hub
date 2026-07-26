import pytest

torch = pytest.importorskip("torch")


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
