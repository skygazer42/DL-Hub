from __future__ import annotations

from dataclasses import dataclass

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
        b, c, h, w = feat.shape
        tokens = feat.view(b, c, h * w).transpose(1, 2)
        return self.proj(tokens)


class AdditiveCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.w_vis = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.w_dec = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.v = nn.Linear(int(hidden_dim), 1, bias=False)

    def forward(
        self, *, visual_tokens: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, h = visual_tokens.shape
        dec = self.w_dec(query).view(b, 1, h).expand(b, n, h)
        scores = self.v(torch.tanh(self.w_vis(visual_tokens) + dec)).squeeze(-1)
        attn = torch.softmax(scores, dim=1)
        context = torch.bmm(attn.unsqueeze(1), visual_tokens).squeeze(1)
        return context, attn


class ITMHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    max_text_length: int
    hidden_dim: int = 64
    vision_width: int = 32
    embed_dim: int = 32


class ToyBLIPModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(
            vision_width=int(cfg.vision_width),
            hidden_dim=int(cfg.hidden_dim),
        )
        self.token_embed = nn.Embedding(
            int(cfg.vocab_size), int(cfg.embed_dim), padding_idx=int(cfg.pad_id)
        )
        self.init_proj = nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim))
        self.attn = AdditiveCrossAttention(int(cfg.hidden_dim))
        self.decoder_cell = nn.GRUCell(int(cfg.embed_dim) + int(cfg.hidden_dim), int(cfg.hidden_dim))
        self.fusion = nn.Linear(int(cfg.hidden_dim) * 2, int(cfg.hidden_dim))
        self.caption_head = nn.Linear(int(cfg.hidden_dim), int(cfg.vocab_size))
        self.itm_head = ITMHead(int(cfg.hidden_dim))

    def _decode(
        self,
        *,
        visual_tokens: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = token_ids.to(torch.long)
        attention_mask = attention_mask.to(torch.float32)
        b, t = token_ids.shape
        hidden = torch.tanh(self.init_proj(visual_tokens.mean(dim=1)))

        fused_states = torch.empty((b, t, int(self.cfg.hidden_dim)), device=token_ids.device)
        attn_map = torch.empty((b, t, int(visual_tokens.shape[1])), device=token_ids.device)

        for step in range(int(t)):
            emb = self.token_embed(token_ids[:, step])
            context, weights = self.attn(visual_tokens=visual_tokens, query=hidden)
            hidden = self.decoder_cell(torch.cat([emb, context], dim=1), hidden)
            fused = torch.tanh(self.fusion(torch.cat([hidden, context], dim=1)))
            fused = fused * attention_mask[:, step].unsqueeze(1)
            fused_states[:, step, :] = fused
            attn_map[:, step, :] = weights
        return fused_states, attn_map

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual_tokens = self.vision_encoder(batch["image"])

        caption_states, caption_attn = self._decode(
            visual_tokens=visual_tokens,
            token_ids=batch["caption_in_ids"],
            attention_mask=batch["caption_mask"],
        )
        itm_states, itm_attn = self._decode(
            visual_tokens=visual_tokens,
            token_ids=batch["itm_input_ids"],
            attention_mask=batch["itm_attention_mask"],
        )
        caption_logits = self.caption_head(caption_states)

        itm_mask = batch["itm_attention_mask"].to(torch.float32).unsqueeze(-1)
        itm_pooled = (itm_states * itm_mask).sum(dim=1) / itm_mask.sum(dim=1).clamp_min(1.0)
        itm_logits = self.itm_head(itm_pooled)

        return {
            "caption_logits": caption_logits,
            "itm_logits": itm_logits,
            "fused_states": itm_states,
            "caption_attn": caption_attn,
            "itm_attn": itm_attn,
        }

    @torch.no_grad()
    def greedy_generate(self, image: torch.Tensor, *, max_length: int) -> torch.Tensor:
        visual_tokens = self.vision_encoder(image)
        b = int(image.shape[0])
        hidden = torch.tanh(self.init_proj(visual_tokens.mean(dim=1)))
        cur = torch.full(
            (b,),
            fill_value=int(self.cfg.bos_id),
            device=image.device,
            dtype=torch.long,
        )
        outputs = torch.full(
            (b, int(max_length)),
            fill_value=int(self.cfg.pad_id),
            device=image.device,
            dtype=torch.long,
        )

        for step in range(int(max_length)):
            emb = self.token_embed(cur)
            context, _ = self.attn(visual_tokens=visual_tokens, query=hidden)
            hidden = self.decoder_cell(torch.cat([emb, context], dim=1), hidden)
            fused = torch.tanh(self.fusion(torch.cat([hidden, context], dim=1)))
            logits = self.caption_head(fused)
            cur = logits.argmax(dim=-1)
            outputs[:, step] = cur
        return outputs


def blip_lite_loss(
    *,
    caption_logits: torch.Tensor,
    itm_logits: torch.Tensor,
    caption_targets: torch.Tensor,
    caption_mask: torch.Tensor,
    itm_targets: torch.Tensor,
    pad_id: int,
    itm_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    del caption_mask
    caption_loss = F.cross_entropy(
        caption_logits.view(-1, caption_logits.shape[-1]),
        caption_targets.view(-1).to(torch.long),
        ignore_index=int(pad_id),
    )
    itm_loss = F.cross_entropy(itm_logits, itm_targets.to(torch.long))
    total = caption_loss + float(itm_weight) * itm_loss
    return {"loss": total, "caption_loss": caption_loss, "itm_loss": itm_loss}


@torch.no_grad()
def token_accuracy(
    caption_logits: torch.Tensor, caption_targets: torch.Tensor, caption_mask: torch.Tensor
) -> float:
    pred = caption_logits.argmax(dim=-1)
    mask = caption_mask.to(torch.bool)
    correct = ((pred == caption_targets) & mask).sum().item()
    total = mask.sum().item()
    if total == 0:
        return 0.0
    return float(correct) / float(total)


@torch.no_grad()
def caption_exact_match(
    caption_logits: torch.Tensor, caption_targets: torch.Tensor, caption_mask: torch.Tensor
) -> float:
    pred = caption_logits.argmax(dim=-1)
    mask = caption_mask.to(torch.bool)
    exact = ((pred == caption_targets) | (~mask)).all(dim=1).to(torch.float32).mean()
    return float(exact.item())


__all__ = [
    "AdditiveCrossAttention",
    "ITMHead",
    "ModelConfig",
    "ToyBLIPModel",
    "VisionEncoder",
    "blip_lite_loss",
    "caption_exact_match",
    "token_accuracy",
]
