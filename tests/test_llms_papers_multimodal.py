import pytest

torch = pytest.importorskip("torch")


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
