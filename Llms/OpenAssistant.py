from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OpenAssistantMessage:
    message_id: str
    role: str
    text: str
    children: tuple[OpenAssistantMessage, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OpenAssistantPreference:
    chosen_id: str
    rejected_id: str
    score_gap: float


@dataclass(frozen=True)
class OpenAssistantPreferenceExample:
    context_messages: tuple[OpenAssistantMessage, ...]
    chosen_message: OpenAssistantMessage
    rejected_message: OpenAssistantMessage
    score_gap: float


@dataclass(frozen=True)
class OpenAssistantConfig:
    languages: int = 35
    messages: int = 161_443
    quality_ratings: int = 461_292
    complete_trees: int = 10_000
    volunteers: int = 13_500


class OpenAssistantConversationTree:
    def __init__(self, *, root: OpenAssistantMessage) -> None:
        self.root = root

    def flatten_messages(self) -> tuple[OpenAssistantMessage, ...]:
        ordered: list[OpenAssistantMessage] = []

        def visit(node: OpenAssistantMessage) -> None:
            ordered.append(node)
            for child in node.children:
                visit(child)

        visit(self.root)
        return tuple(ordered)

    def message_by_id(self, message_id: str) -> OpenAssistantMessage:
        for message in self.flatten_messages():
            if message.message_id == message_id:
                return message
        raise KeyError(f"unknown message id: {message_id}")

    def path_to(self, message_id: str) -> tuple[OpenAssistantMessage, ...]:
        path: list[OpenAssistantMessage] = []

        def visit(node: OpenAssistantMessage) -> bool:
            path.append(node)
            if node.message_id == message_id:
                return True
            for child in node.children:
                if visit(child):
                    return True
            path.pop()
            return False

        if not visit(self.root):
            raise KeyError(f"unknown message id: {message_id}")
        return tuple(path)

    def root_to_leaf_paths(self) -> tuple[tuple[OpenAssistantMessage, ...], ...]:
        paths: list[tuple[OpenAssistantMessage, ...]] = []

        def visit(node: OpenAssistantMessage, path: tuple[OpenAssistantMessage, ...]) -> None:
            next_path = path + (node,)
            if not node.children:
                paths.append(next_path)
                return
            for child in node.children:
                visit(child, next_path)

        visit(self.root, ())
        return tuple(paths)


class OpenAssistantDataset:
    def __init__(
        self,
        config: OpenAssistantConfig,
        *,
        trees: tuple[OpenAssistantConversationTree, ...] = (),
        preferences: tuple[OpenAssistantPreference, ...] = (),
    ) -> None:
        self.config = config
        self.trees = tuple(trees)
        self.preferences = tuple(preferences)

    def preference_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple((pref.chosen_id, pref.rejected_id) for pref in self.preferences)

    def flattened_messages(self) -> tuple[OpenAssistantMessage, ...]:
        flattened: list[OpenAssistantMessage] = []
        for tree in self.trees:
            flattened.extend(tree.flatten_messages())
        return tuple(flattened)

    def message_by_id(self, message_id: str) -> OpenAssistantMessage:
        for tree in self.trees:
            try:
                return tree.message_by_id(message_id)
            except KeyError:
                continue
        raise KeyError(f"unknown message id: {message_id}")

    def preference_examples(self) -> tuple[OpenAssistantPreferenceExample, ...]:
        examples: list[OpenAssistantPreferenceExample] = []
        for preference in self.preferences:
            chosen_tree, chosen_path = self._tree_and_path_for_message(preference.chosen_id)
            rejected_tree, rejected_path = self._tree_and_path_for_message(preference.rejected_id)
            if chosen_tree is not rejected_tree:
                raise ValueError("preference pairs must resolve within the same conversation tree")

            prefix_length = 0
            for chosen_message, rejected_message in zip(chosen_path, rejected_path):
                if chosen_message.message_id != rejected_message.message_id:
                    break
                prefix_length += 1

            examples.append(
                OpenAssistantPreferenceExample(
                    context_messages=chosen_path[:prefix_length],
                    chosen_message=chosen_path[-1],
                    rejected_message=rejected_path[-1],
                    score_gap=float(preference.score_gap),
                )
            )
        return tuple(examples)

    def _tree_and_path_for_message(
        self,
        message_id: str,
    ) -> tuple[OpenAssistantConversationTree, tuple[OpenAssistantMessage, ...]]:
        for tree in self.trees:
            try:
                return tree, tree.path_to(message_id)
            except KeyError:
                continue
        raise KeyError(f"unknown message id: {message_id}")


__all__ = [
    "OpenAssistantConfig",
    "OpenAssistantConversationTree",
    "OpenAssistantDataset",
    "OpenAssistantMessage",
    "OpenAssistantPreference",
    "OpenAssistantPreferenceExample",
]
