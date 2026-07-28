from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.synthetic_text import simple_tokenize


@dataclass(frozen=True)
class TagVocab:
    tag_to_id: dict[str, int]
    id_to_tag: list[str]

    @property
    def size(self) -> int:
        return len(self.id_to_tag)

    def to_dict(self) -> dict[str, object]:
        return {"tag_to_id": dict(self.tag_to_id)}


def _make_tag_vocab() -> TagVocab:
    tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
    return TagVocab(tag_to_id={t: i for i, t in enumerate(tags)}, id_to_tag=list(tags))


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_id: int
    unk_id: int

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "token_to_id": dict(self.token_to_id),
        }


def _build_vocab(token_sequences: list[list[str]]) -> Vocab:
    tokens: list[str] = []
    for seq in token_sequences:
        tokens.extend(seq)

    id_to_token = ["<pad>", "<unk>"]
    token_to_id = {"<pad>": 0, "<unk>": 1}

    for tok in sorted(set(tokens)):
        if tok in token_to_id:
            continue
        token_to_id[tok] = len(id_to_token)
        id_to_token.append(tok)

    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 64
    max_length: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    ignore_index: int = -100


def _bio_tags(entity_type: str, n_tokens: int) -> list[str]:
    if n_tokens <= 0:
        raise ValueError("n_tokens must be > 0")
    if n_tokens == 1:
        return [f"B-{entity_type}"]
    return [f"B-{entity_type}"] + [f"I-{entity_type}"] * (n_tokens - 1)


def _make_synthetic_ner_examples(num_samples: int, seed: int) -> list[tuple[list[str], list[str]]]:
    rng = np.random.default_rng(int(seed))

    persons = [["alice"], ["bob"], ["carol"], ["dave"], ["john"], ["mary"]]
    locations = [["paris"], ["london"], ["beijing"], ["new", "york"], ["san", "francisco"]]
    orgs = [["openai"], ["google"], ["microsoft"], ["acme", "corp"]]

    templates = [
        "i met {PER} in {LOC}",
        "{PER} lives in {LOC}",
        "{PER} works at {ORG}",
        "{ORG} is in {LOC}",
        "{PER} moved to {LOC}",
    ]

    examples: list[tuple[list[str], list[str]]] = []
    num_samples = int(num_samples)
    for _ in range(num_samples):
        tmpl = templates[int(rng.integers(0, len(templates)))]
        per = persons[int(rng.integers(0, len(persons)))]
        loc = locations[int(rng.integers(0, len(locations)))]
        org = orgs[int(rng.integers(0, len(orgs)))]

        # Build a token-level sequence and BIO tag sequence.
        tokens: list[str] = []
        tags: list[str] = []

        # A tiny trick: expand the template using placeholders, then re-tokenize with the shared tokenizer.
        text = tmpl.format(PER=" ".join(per), LOC=" ".join(loc), ORG=" ".join(org))
        tokens = simple_tokenize(text)

        # Label entities by scanning exact token spans (since we control generation).
        tags = ["O"] * len(tokens)

        def _mark(entity_tokens: list[str], entity_type: str) -> None:
            if not entity_tokens:
                return
            for i in range(0, len(tokens) - len(entity_tokens) + 1):
                if tokens[i : i + len(entity_tokens)] == entity_tokens:
                    bio = _bio_tags(entity_type, len(entity_tokens))
                    tags[i : i + len(entity_tokens)] = bio
                    return

        _mark(per, "PER")
        _mark(loc, "LOC")
        _mark(org, "ORG")

        examples.append((tokens, tags))

    rng.shuffle(examples)
    return examples


class SyntheticNerDataset:
    def __init__(
        self,
        *,
        token_seqs: list[list[str]],
        tag_seqs: list[list[str]],
        vocab: Vocab,
        tag_vocab: TagVocab,
        max_length: int,
        ignore_index: int,
    ) -> None:
        self.vocab = vocab
        self.tag_vocab = tag_vocab
        self.max_length = int(max_length)
        self.ignore_index = int(ignore_index)

        self.token_seqs = token_seqs
        self.tag_seqs = tag_seqs

        if len(self.token_seqs) != len(self.tag_seqs):
            raise ValueError("token_seqs and tag_seqs must have the same length")

    def __len__(self) -> int:
        return len(self.token_seqs)

    def __getitem__(self, idx: int):
        import torch

        tokens = self.token_seqs[int(idx)][: self.max_length]
        tags = self.tag_seqs[int(idx)][: self.max_length]
        if len(tokens) != len(tags):
            raise RuntimeError("token/tag length mismatch")

        input_ids = [self.vocab.token_to_id.get(t, self.vocab.unk_id) for t in tokens]
        attention_mask = [1] * len(input_ids)
        label_ids = [self.tag_vocab.tag_to_id[t] for t in tags]

        while len(input_ids) < self.max_length:
            input_ids.append(self.vocab.pad_id)
            attention_mask.append(0)
            label_ids.append(self.ignore_index)

        inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }
        labels = torch.tensor(label_ids, dtype=torch.long)
        return inputs, labels


def get_dataloaders(config: DataConfig):
    from torch.utils.data import DataLoader, Subset

    tag_vocab = _make_tag_vocab()
    examples = _make_synthetic_ner_examples(num_samples=config.num_samples, seed=config.seed)
    token_seqs = [t for t, _ in examples]
    tag_seqs = [y for _, y in examples]

    vocab = _build_vocab(token_seqs)
    ds = SyntheticNerDataset(
        token_seqs=token_seqs,
        tag_seqs=tag_seqs,
        vocab=vocab,
        tag_vocab=tag_vocab,
        max_length=config.max_length,
        ignore_index=config.ignore_index,
    )

    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(config.val_fraction), seed=int(config.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
    )
    return train_loader, val_loader, vocab, tag_vocab


__all__ = ["DataConfig", "TagVocab", "SyntheticNerDataset", "Vocab", "get_dataloaders"]
