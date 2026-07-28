from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class MultiViewVisionEncoder(nn.Module):
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views = int(images.shape[0]), int(images.shape[1])
        flat = images.view(batch_size * num_views, *images.shape[2:]).to(torch.float32)
        feat = self.features(flat)
        _, channels, height, width = feat.shape
        tokens = feat.view(batch_size * num_views, channels, height * width).transpose(1, 2)
        tokens = self.proj(tokens)
        return tokens.view(batch_size, num_views * height * width, int(tokens.shape[-1]))


class PerceiverResamplerBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.latent_norm1 = nn.LayerNorm(int(hidden_dim))
        self.self_attn = nn.MultiheadAttention(
            embed_dim=int(hidden_dim),
            num_heads=1,
            batch_first=True,
        )
        self.latent_norm2 = nn.LayerNorm(int(hidden_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=int(hidden_dim),
            num_heads=1,
            batch_first=True,
        )
        self.latent_norm3 = nn.LayerNorm(int(hidden_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim) * 2),
            nn.ReLU(),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim)),
        )

    def forward(self, latents: torch.Tensor, visual_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self_in = self.latent_norm1(latents)
        self_out, _ = self.self_attn(self_in, self_in, self_in, need_weights=False)
        latents = latents + self_out

        cross_in = self.latent_norm2(latents)
        cross_out, attn = self.cross_attn(cross_in, visual_tokens, visual_tokens, need_weights=True)
        latents = latents + cross_out

        ff_in = self.latent_norm3(latents)
        latents = latents + self.ff(ff_in)
        return latents, attn


class PerceiverResampler(nn.Module):
    def __init__(self, *, hidden_dim: int, num_latents: int, num_layers: int = 2) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(int(num_latents), int(hidden_dim)) * 0.02)
        self.layers = nn.ModuleList(
            [PerceiverResamplerBlock(int(hidden_dim)) for _ in range(int(num_layers))]
        )

    def forward(self, visual_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.latents.unsqueeze(0).expand(int(visual_tokens.shape[0]), -1, -1)
        last_attn = torch.empty(0, device=visual_tokens.device)
        for layer in self.layers:
            latents, last_attn = layer(latents, visual_tokens)
        return latents, last_attn


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
        resampled_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_emb = self.token_proj(self.token_embed(input_ids.to(torch.long)))
        text_emb = text_emb * attention_mask.to(torch.float32).unsqueeze(-1)
        fused = torch.cat([resampled_tokens, text_emb], dim=1)
        out, _ = self.decoder(fused)
        text_out = out[:, int(resampled_tokens.shape[1]) :, :]
        return self.head(self.norm(text_out))


@dataclass(frozen=True)
class PerceiverModelConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    sep_id: int
    max_text_length: int
    hidden_dim: int = 64
    vision_width: int = 32
    embed_dim: int = 32
    num_latents: int = 6


class CompactPerceiverResamplerModel(nn.Module):
    def __init__(self, cfg: PerceiverModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = MultiViewVisionEncoder(
            vision_width=int(cfg.vision_width),
            hidden_dim=int(cfg.hidden_dim),
        )
        self.resampler = PerceiverResampler(
            hidden_dim=int(cfg.hidden_dim),
            num_latents=int(cfg.num_latents),
        )
        self.decoder_lm = TinyDecoderLM(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            hidden_dim=int(cfg.hidden_dim),
            embed_dim=int(cfg.embed_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual_tokens = self.vision_encoder(batch["images"])
        resampled_tokens, attn = self.resampler(visual_tokens)
        logits = self.decoder_lm(
            resampled_tokens=resampled_tokens,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return {
            "logits": logits,
            "resampled_tokens": resampled_tokens,
            "resampler_attn": attn,
        }

    @torch.no_grad()
    def greedy_generate(
        self,
        *,
        images: torch.Tensor,
        question_ids: torch.Tensor,
        max_new_tokens: int = 3,
    ) -> torch.Tensor:
        cur = question_ids.clone().to(torch.long)
        lengths = (cur != int(self.cfg.pad_id)).to(torch.long).sum(dim=1)
        done = torch.zeros(int(cur.shape[0]), device=cur.device, dtype=torch.bool)

        for _ in range(int(max_new_tokens)):
            mask = (cur != int(self.cfg.pad_id)).to(torch.float32)
            outputs = self({"images": images, "input_ids": cur, "attention_mask": mask})
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
    "MultiViewVisionEncoder",
    "PerceiverModelConfig",
    "PerceiverResampler",
    "PerceiverResamplerBlock",
    "TinyDecoderLM",
    "CompactPerceiverResamplerModel",
    "answer_exact_match",
    "answer_token_accuracy",
    "qa_loss",
]
