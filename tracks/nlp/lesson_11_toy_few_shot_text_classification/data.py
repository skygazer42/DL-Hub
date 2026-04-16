from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.toy_text import Vocab, simple_tokenize


@dataclass(frozen=True)
class DataConfig:
    num_episodes: int = 512
    batch_size: int = 8
    num_ways: int = 3
    shots: int = 2
    queries_per_class: int = 3
    max_length: int = 12
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


@dataclass(frozen=True)
class EpisodeSpec:
    support_texts: tuple[str, ...]
    support_labels: tuple[int, ...]
    query_texts: tuple[str, ...]
    query_labels: tuple[int, ...]
    class_names: tuple[str, ...]


_INTENTS = (
    {
        "name": "weather",
        "keyword": "rain",
        "context": "forecast",
        "action": "umbrella",
        "slot": "outside",
    },
    {
        "name": "navigation",
        "keyword": "route",
        "context": "traffic",
        "action": "map",
        "slot": "downtown",
    },
    {
        "name": "music",
        "keyword": "playlist",
        "context": "melody",
        "action": "speaker",
        "slot": "studio",
    },
    {
        "name": "booking",
        "keyword": "ticket",
        "context": "travel",
        "action": "hotel",
        "slot": "airport",
    },
    {
        "name": "reminder",
        "keyword": "alarm",
        "context": "calendar",
        "action": "schedule",
        "slot": "morning",
    },
    {
        "name": "lighting",
        "keyword": "lamp",
        "context": "brightness",
        "action": "switch",
        "slot": "livingroom",
    },
    {
        "name": "translation",
        "keyword": "phrase",
        "context": "language",
        "action": "dictionary",
        "slot": "travelguide",
    },
    {
        "name": "finance",
        "keyword": "budget",
        "context": "expense",
        "action": "report",
        "slot": "monthly",
    },
)

_SUPPORT_TEMPLATES = (
    "intent {name} uses {keyword} with {context} and {action}",
    "{name} intent example keeps {keyword} near {slot} for {context}",
    "support intent {name} asks for {action} when {keyword} appears",
)

_QUERY_TEMPLATES = (
    "query intent {name} needs {action} after {keyword} in {context}",
    "which intent handles {keyword} with {context} around {slot}",
    "user request about {name} wants {action} plus {keyword}",
)


def _build_vocab(texts: list[str]) -> Vocab:
    token_to_id = {"<pad>": 0, "<unk>": 1}
    id_to_token = ["<pad>", "<unk>"]
    for text in texts:
        for token in simple_tokenize(text):
            if token in token_to_id:
                continue
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)


def _encode(vocab: Vocab, text: str, *, max_length: int) -> tuple[list[int], list[int]]:
    tokens = simple_tokenize(text)
    ids = [vocab.token_to_id.get(token, vocab.unk_id) for token in tokens[: int(max_length)]]
    mask = [1] * len(ids)
    while len(ids) < int(max_length):
        ids.append(vocab.pad_id)
        mask.append(0)
    return ids, mask


def _render_text(concept: dict[str, str], template_pool: tuple[str, ...], rng: np.random.Generator) -> str:
    template = str(rng.choice(template_pool))
    return template.format(**concept)


def _shuffle_by_permutation(items: list[str], labels: list[int], rng: np.random.Generator) -> tuple[list[str], list[int]]:
    order = rng.permutation(len(items))
    return [items[int(idx)] for idx in order], [labels[int(idx)] for idx in order]


def _make_episode_specs(config: DataConfig) -> list[EpisodeSpec]:
    rng = np.random.default_rng(int(config.seed))
    specs: list[EpisodeSpec] = []
    for _ in range(int(config.num_episodes)):
        chosen = rng.choice(len(_INTENTS), size=int(config.num_ways), replace=False)
        support_texts: list[str] = []
        support_labels: list[int] = []
        query_texts: list[str] = []
        query_labels: list[int] = []
        class_names: list[str] = []

        for class_id, concept_idx in enumerate(chosen):
            concept = dict(_INTENTS[int(concept_idx)])
            class_names.append(concept["name"])

            for _shot in range(int(config.shots)):
                support_texts.append(_render_text(concept, _SUPPORT_TEMPLATES, rng))
                support_labels.append(class_id)
            for _query in range(int(config.queries_per_class)):
                query_texts.append(_render_text(concept, _QUERY_TEMPLATES, rng))
                query_labels.append(class_id)

        support_texts, support_labels = _shuffle_by_permutation(support_texts, support_labels, rng)
        query_texts, query_labels = _shuffle_by_permutation(query_texts, query_labels, rng)
        specs.append(
            EpisodeSpec(
                support_texts=tuple(support_texts),
                support_labels=tuple(support_labels),
                query_texts=tuple(query_texts),
                query_labels=tuple(query_labels),
                class_names=tuple(class_names),
            )
        )
    return specs


class EpisodeDataset:
    def __init__(self, *, episodes: list[EpisodeSpec], vocab: Vocab, max_length: int) -> None:
        self.episodes = list(episodes)
        self.vocab = vocab
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int):
        import torch

        episode = self.episodes[int(idx)]
        support_ids: list[list[int]] = []
        support_mask: list[list[int]] = []
        query_ids: list[list[int]] = []
        query_mask: list[list[int]] = []

        for text in episode.support_texts:
            ids, mask = _encode(self.vocab, text, max_length=self.max_length)
            support_ids.append(ids)
            support_mask.append(mask)
        for text in episode.query_texts:
            ids, mask = _encode(self.vocab, text, max_length=self.max_length)
            query_ids.append(ids)
            query_mask.append(mask)

        return {
            "support_input_ids": torch.tensor(support_ids, dtype=torch.long),
            "support_attention_mask": torch.tensor(support_mask, dtype=torch.float32),
            "support_labels": torch.tensor(episode.support_labels, dtype=torch.long),
            "query_input_ids": torch.tensor(query_ids, dtype=torch.long),
            "query_attention_mask": torch.tensor(query_mask, dtype=torch.float32),
            "query_labels": torch.tensor(episode.query_labels, dtype=torch.long),
            "class_names": list(episode.class_names),
        }


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    episodes = _make_episode_specs(config)
    vocab_texts: list[str] = []
    for concept in _INTENTS:
        for template in _SUPPORT_TEMPLATES + _QUERY_TEMPLATES:
            vocab_texts.append(template.format(**concept))
    vocab = _build_vocab(vocab_texts)
    dataset = EpisodeDataset(episodes=episodes, vocab=vocab, max_length=config.max_length)

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(config.val_fraction),
        seed=int(config.seed),
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    def _collate(batch):
        support_input_ids = torch.stack([item["support_input_ids"] for item in batch], dim=0)
        support_attention_mask = torch.stack([item["support_attention_mask"] for item in batch], dim=0)
        support_labels = torch.stack([item["support_labels"] for item in batch], dim=0)
        query_input_ids = torch.stack([item["query_input_ids"] for item in batch], dim=0)
        query_attention_mask = torch.stack([item["query_attention_mask"] for item in batch], dim=0)
        query_labels = torch.stack([item["query_labels"] for item in batch], dim=0)
        class_names = [item["class_names"] for item in batch]
        return {
            "support_input_ids": support_input_ids,
            "support_attention_mask": support_attention_mask,
            "support_labels": support_labels,
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "query_labels": query_labels,
            "class_names": class_names,
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab


__all__ = ["DataConfig", "EpisodeDataset", "Vocab", "get_dataloaders"]
