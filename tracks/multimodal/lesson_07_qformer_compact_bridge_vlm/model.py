from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn


class VisionEncoder(nn.Module):
    def __init__(self, *, vision_width: int, hidden_dim: int) -> None:
        super().__init__()
        mid = max(16, int(vision_width) // 2)
        self.features = nn.Sequential(
            nn.Conv2d(3, mid, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid, int(vision_width), kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(int(vision_width), int(hidden_dim))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.features(image.to(torch.float32))
        batch_size, channels, height, width = feat.shape
        tokens = feat.view(batch_size, channels, height * width).transpose(1, 2)
        return self.proj(tokens)


class QueryCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.k_proj = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.v_proj = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.out_proj = nn.Linear(int(hidden_dim), int(hidden_dim))

    def forward(
        self, *, query_states: torch.Tensor, visual_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.q_proj(query_states)
        key = self.k_proj(visual_tokens)
        value = self.v_proj(visual_tokens)
        scale = math.sqrt(float(query.shape[-1]))
        scores = torch.matmul(query, key.transpose(1, 2)) / scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        return self.out_proj(context), attn


class QFormerBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = QueryCrossAttention(int(hidden_dim))
        self.norm1 = nn.LayerNorm(int(hidden_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim) * 2),
            nn.ReLU(),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
        )
        self.norm2 = nn.LayerNorm(int(hidden_dim))

    def forward(
        self, *, query_states: torch.Tensor, visual_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, attn = self.attn(query_states=query_states, visual_tokens=visual_tokens)
        query_states = self.norm1(query_states + context)
        query_states = self.norm2(query_states + self.ff(query_states))
        return query_states, attn


class QFormerBridge(nn.Module):
    def __init__(self, *, hidden_dim: int, num_query_tokens: int, num_layers: int = 2) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(int(num_query_tokens), int(hidden_dim)) * 0.02)
        self.layers = nn.ModuleList([QFormerBlock(int(hidden_dim)) for _ in range(int(num_layers))])

    def forward(self, visual_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_states = self.query_tokens.unsqueeze(0).expand(int(visual_tokens.shape[0]), -1, -1)
        last_attn = torch.empty(0, device=visual_tokens.device)
        for layer in self.layers:
            query_states, last_attn = layer(query_states=query_states, visual_tokens=visual_tokens)
        return query_states, last_attn


class TinyDecoderLM(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, hidden_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.token_embed = nn.Embedding(int(vocab_size), int(embed_dim), padding_idx=int(pad_id))
        self.token_proj = nn.Linear(int(embed_dim), int(hidden_dim))
        self.decoder = nn.GRU(int(hidden_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.head = nn.Linear(int(hidden_dim), int(vocab_size))

    def forward(
        self,
        *,
        query_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_emb = self.token_proj(self.token_embed(input_ids.to(torch.long)))
        text_emb = text_emb * attention_mask.to(torch.float32).unsqueeze(-1)
        fused = torch.cat([query_states, text_emb], dim=1)
        out, _ = self.decoder(fused)
        text_out = out[:, int(query_states.shape[1]) :, :]
        return self.head(self.norm(text_out))


@dataclass(frozen=True)
class QFormerModelConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    sep_id: int
    max_text_length: int
    hidden_dim: int = 64
    vision_width: int = 32
    embed_dim: int = 32
    num_query_tokens: int = 4


class CompactQFormerModel(nn.Module):
    def __init__(self, cfg: QFormerModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(
            vision_width=int(cfg.vision_width),
            hidden_dim=int(cfg.hidden_dim),
        )
        self.qformer = QFormerBridge(
            hidden_dim=int(cfg.hidden_dim),
            num_query_tokens=int(cfg.num_query_tokens),
        )
        self.decoder_lm = TinyDecoderLM(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            hidden_dim=int(cfg.hidden_dim),
            embed_dim=int(cfg.embed_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual_tokens = self.vision_encoder(batch["image"])
        query_states, attn = self.qformer(visual_tokens)
        logits = self.decoder_lm(
            query_states=query_states,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return {
            "logits": logits,
            "query_states": query_states,
            "query_attn": attn,
        }

    @torch.no_grad()
    def greedy_generate(
        self,
        *,
        image: torch.Tensor,
        question_ids: torch.Tensor,
        max_new_tokens: int = 3,
    ) -> torch.Tensor:
        cur = question_ids.clone().to(torch.long)
        lengths = (cur != int(self.cfg.pad_id)).to(torch.long).sum(dim=1)
        done = torch.zeros(int(cur.shape[0]), device=cur.device, dtype=torch.bool)

        for _ in range(int(max_new_tokens)):
            mask = (cur != int(self.cfg.pad_id)).to(torch.float32)
            outputs = self({"image": image, "input_ids": cur, "attention_mask": mask})
            last_pos = (lengths - 1).clamp_min(0)
            next_logits = outputs["logits"][torch.arange(int(cur.shape[0]), device=cur.device), last_pos]
            next_token = next_logits.argmax(dim=-1)

            for row in range(int(cur.shape[0])):
                if done[row]:
                    continue
                pos = int(lengths[row].item())
                if pos >= int(cur.shape[1]):
                    done[row] = True
                    continue
                cur[row, pos] = next_token[row]
                lengths[row] = lengths[row] + 1
                if int(next_token[row].item()) == int(self.cfg.eos_id):
                    done[row] = True
            if bool(done.all()):
                break
        return cur


def qa_loss(logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1).to(torch.long),
        ignore_index=int(ignore_index),
    )


@torch.no_grad()
def answer_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100) -> float:
    pred = logits.argmax(dim=-1)
    mask = labels != int(ignore_index)
    total = mask.sum().item()
    if total == 0:
        return 0.0
    correct = ((pred == labels) & mask).sum().item()
    return float(correct) / float(total)


@torch.no_grad()
def answer_exact_match(logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100) -> float:
    pred = logits.argmax(dim=-1)
    mask = labels != int(ignore_index)
    exact = ((pred == labels) | (~mask)).all(dim=1).to(torch.float32).mean()
    return float(exact.item())


__all__ = [
    "QFormerBlock",
    "QFormerBridge",
    "QFormerModelConfig",
    "QueryCrossAttention",
    "TinyDecoderLM",
    "CompactQFormerModel",
    "VisionEncoder",
    "answer_exact_match",
    "answer_token_accuracy",
    "qa_loss",
]
