from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TinyDocVisionEncoder(nn.Module):
    def __init__(self, vision_width: int) -> None:
        super().__init__()
        mid = max(16, int(vision_width) // 2)
        self.features = nn.Sequential(
            nn.Conv2d(3, mid, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid, int(vision_width), kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(vision_width), int(vision_width), kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(int(vision_width))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.features(image.to(torch.float32))
        batch_size, channels, height, width = feat.shape
        tokens = feat.view(batch_size, channels, height * width).transpose(1, 2)
        return self.norm(tokens)


class VisionProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(in_dim), int(out_dim))

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(visual_tokens)


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
        visual_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_emb = self.token_proj(self.token_embed(input_ids.to(torch.long)))
        text_emb = text_emb * attention_mask.to(torch.float32).unsqueeze(-1)
        fused = torch.cat([visual_tokens, text_emb], dim=1)
        out, _ = self.decoder(fused)
        text_out = out[:, int(visual_tokens.shape[1]) :, :]
        return self.head(self.norm(text_out))


@dataclass(frozen=True)
class DocOcrModelConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    sep_id: int
    max_text_length: int
    hidden_dim: int = 64
    vision_width: int = 32
    embed_dim: int = 32


class ToyDocOcrModel(nn.Module):
    def __init__(self, cfg: DocOcrModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = TinyDocVisionEncoder(int(cfg.vision_width))
        self.vision_projector = VisionProjector(int(cfg.vision_width), int(cfg.hidden_dim))
        self.decoder_lm = TinyDecoderLM(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            hidden_dim=int(cfg.hidden_dim),
            embed_dim=int(cfg.embed_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        visual_raw = self.vision_encoder(batch["image"])
        visual_tokens = self.vision_projector(visual_raw)
        logits = self.decoder_lm(
            visual_tokens=visual_tokens,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return {"logits": logits, "visual_tokens": visual_tokens}

    @torch.no_grad()
    def greedy_generate(
        self,
        *,
        image: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 3,
    ) -> torch.Tensor:
        cur = prompt_ids.clone().to(torch.long)
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


def ocr_loss(logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100) -> torch.Tensor:
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


@torch.no_grad()
def present_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    none_id: int,
    ignore_index: int = -100,
) -> float:
    pred = logits.argmax(dim=-1)
    valid = labels != int(ignore_index)
    first_valid = valid.to(torch.long).argmax(dim=1)
    row_idx = torch.arange(int(labels.shape[0]), device=labels.device)
    pred_first = pred[row_idx, first_valid]
    label_first = labels[row_idx, first_valid]
    pred_present = pred_first != int(none_id)
    label_present = label_first != int(none_id)
    return float((pred_present == label_present).to(torch.float32).mean().item())


__all__ = [
    "DocOcrModelConfig",
    "TinyDecoderLM",
    "TinyDocVisionEncoder",
    "ToyDocOcrModel",
    "VisionProjector",
    "answer_exact_match",
    "answer_token_accuracy",
    "ocr_loss",
    "present_accuracy",
]
