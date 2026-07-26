import pytest

torch = pytest.importorskip("torch")


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
