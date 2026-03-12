from __future__ import annotations

from dataclasses import dataclass

from .openassistant import (
    OpenAssistantConfig,
    OpenAssistantConversationTree,
    OpenAssistantDataset,
    OpenAssistantMessage,
    OpenAssistantPreference,
    OpenAssistantPreferenceExample,
)


@dataclass(frozen=True)
class OASST1Config:
    dataset_name: str = "OpenAssistant Conversations Dataset"
    alias_of: str = "OpenAssistant"
    languages: int = 35


class OASST1Dataset:
    def __init__(
        self,
        config: OASST1Config,
        *,
        trees: tuple[OpenAssistantConversationTree, ...] = (),
        preferences: tuple[OpenAssistantPreference, ...] = (),
    ) -> None:
        self.config = config
        self.base_dataset = OpenAssistantDataset(
            OpenAssistantConfig(languages=int(config.languages)),
            trees=trees,
            preferences=preferences,
        )

    def preference_pairs(self) -> tuple[tuple[str, str], ...]:
        return self.base_dataset.preference_pairs()

    def flattened_messages(self) -> tuple[OpenAssistantMessage, ...]:
        return self.base_dataset.flattened_messages()

    def message_by_id(self, message_id: str) -> OpenAssistantMessage:
        return self.base_dataset.message_by_id(message_id)

    def preference_examples(self) -> tuple[OpenAssistantPreferenceExample, ...]:
        return self.base_dataset.preference_examples()


__all__ = [
    "OASST1Config",
    "OASST1Dataset",
]
