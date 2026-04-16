from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.toy_text import Vocab, simple_tokenize


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    batch_size: int = 16
    num_classes: int = 3
    support_per_class: int = 2
    max_length: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


@dataclass(frozen=True)
class InContextSample:
    support_texts: tuple[str, ...]
    support_labels: tuple[int, ...]
    query_text: str
    query_label: int
    prompt_text: str


_INTENTS = (
    {"name": "weather", "keyword": "rain", "context": "forecast", "action": "umbrella"},
    {"name": "navigation", "keyword": "route", "context": "traffic", "action": "map"},
    {"name": "music", "keyword": "playlist", "context": "melody", "action": "speaker"},
    {"name": "booking", "keyword": "ticket", "context": "travel", "action": "hotel"},
    {"name": "finance", "keyword": "budget", "context": "expense", "action": "report"},
)

_SUPPORT_TEMPLATES = (
    "intent {name}: keyword {keyword} with {context} needs {action}",
    "support intent {name} uses {keyword} and {action} in {context}",
    "intent {name} example has {keyword} plus {context} for {action}",
)

_QUERY_TEMPLATES = (
    "query about {keyword} and {context} asks for {action}",
    "user asks intent with {keyword} around {context}",
    "need help with {name} using {keyword} and {action}",
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


def _render_prompt(
    support_texts: list[str], support_labels: list[int], query_text: str, class_names: list[str]
) -> str:
    lines = ["Support examples:"]
    for text, label in zip(support_texts, support_labels):
        lines.append(f"- label={class_names[label]} text={text}")
    lines.append(f"Query: {query_text}")
    lines.append("Predict the intent label.")
    return "\n".join(lines)


def _encode(vocab: Vocab, text: str, *, max_length: int) -> tuple[list[int], list[int]]:
    ids = [vocab.token_to_id.get(tok, vocab.unk_id) for tok in simple_tokenize(text)[: int(max_length)]]
    attn = [1] * len(ids)
    while len(ids) < int(max_length):
        ids.append(vocab.pad_id)
        attn.append(0)
    return ids, attn


def _make_samples(config: DataConfig) -> list[InContextSample]:
    rng = np.random.default_rng(int(config.seed))
    if int(config.num_classes) <= 1 or int(config.num_classes) > len(_INTENTS):
        raise ValueError("num_classes must be in [2, len(_INTENTS)]")

    samples: list[InContextSample] = []
    for _ in range(int(config.num_samples)):
        chosen_indices = rng.choice(len(_INTENTS), size=int(config.num_classes), replace=False)
        chosen_intents = [dict(_INTENTS[int(idx)]) for idx in chosen_indices]
        class_names = [intent["name"] for intent in chosen_intents]

        support_texts: list[str] = []
        support_labels: list[int] = []
        for class_id, intent in enumerate(chosen_intents):
            for _ in range(int(config.support_per_class)):
                template = str(rng.choice(_SUPPORT_TEMPLATES))
                support_texts.append(template.format(**intent))
                support_labels.append(class_id)

        query_label = int(rng.integers(0, int(config.num_classes)))
        query_template = str(rng.choice(_QUERY_TEMPLATES))
        query_text = query_template.format(**chosen_intents[query_label])
        prompt_text = _render_prompt(support_texts, support_labels, query_text, class_names)
        samples.append(
            InContextSample(
                support_texts=tuple(support_texts),
                support_labels=tuple(support_labels),
                query_text=query_text,
                query_label=query_label,
                prompt_text=prompt_text,
            )
        )
    return samples


class InContextTextDataset:
    def __init__(self, *, samples: list[InContextSample], vocab: Vocab, max_length: int) -> None:
        self.samples = list(samples)
        self.vocab = vocab
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        sample = self.samples[int(idx)]
        support_ids: list[list[int]] = []
        support_attn: list[list[int]] = []
        for text in sample.support_texts:
            ids, attn = _encode(self.vocab, text, max_length=self.max_length)
            support_ids.append(ids)
            support_attn.append(attn)

        query_ids, query_attn = _encode(self.vocab, sample.query_text, max_length=self.max_length)
        return {
            "support_input_ids": torch.tensor(support_ids, dtype=torch.long),
            "support_attention_mask": torch.tensor(support_attn, dtype=torch.float32),
            "support_labels": torch.tensor(sample.support_labels, dtype=torch.long),
            "query_input_ids": torch.tensor(query_ids, dtype=torch.long),
            "query_attention_mask": torch.tensor(query_attn, dtype=torch.float32),
            "query_labels": torch.tensor(sample.query_label, dtype=torch.long),
            "prompt_text": sample.prompt_text,
        }


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    samples = _make_samples(config)
    vocab_texts: list[str] = []
    for sample in samples:
        vocab_texts.extend(sample.support_texts)
        vocab_texts.append(sample.query_text)
    vocab = _build_vocab(vocab_texts)

    dataset = InContextTextDataset(samples=samples, vocab=vocab, max_length=config.max_length)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(config.val_fraction),
        seed=int(config.seed),
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    def _collate(batch):
        return {
            "support_input_ids": torch.stack([item["support_input_ids"] for item in batch], dim=0),
            "support_attention_mask": torch.stack([item["support_attention_mask"] for item in batch], dim=0),
            "support_labels": torch.stack([item["support_labels"] for item in batch], dim=0),
            "query_input_ids": torch.stack([item["query_input_ids"] for item in batch], dim=0),
            "query_attention_mask": torch.stack([item["query_attention_mask"] for item in batch], dim=0),
            "query_labels": torch.stack([item["query_labels"] for item in batch], dim=0),
            "prompt_text": [item["prompt_text"] for item in batch],
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


__all__ = ["DataConfig", "InContextTextDataset", "get_dataloaders"]
