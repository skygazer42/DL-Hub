import pytest

torch = pytest.importorskip("torch")


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
