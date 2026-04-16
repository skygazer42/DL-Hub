from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class VisionEncoder(nn.Module):
    def __init__(self, *, width: int) -> None:
        super().__init__()
        hidden = max(16, int(width) // 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, int(width), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (B, C, H, W), got {tuple(image.shape)}")
        return self.net(image.to(torch.float32))


class FrozenTextEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, width: int) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.embedding = nn.Embedding(int(vocab_size), int(width), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(width), int(width))

    def token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids.to(torch.long))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prompt_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embed = self.token_embeddings(input_ids)
        mask = attention_mask.to(torch.float32)
        pieces = [token_embed]
        mask_pieces = [mask]
        if prompt_embed is not None:
            pieces.insert(0, prompt_embed.to(token_embed.dtype))
            prompt_mask = torch.ones(
                int(prompt_embed.shape[0]),
                int(prompt_embed.shape[1]),
                device=token_embed.device,
                dtype=torch.float32,
            )
            mask_pieces.insert(0, prompt_mask)
        merged = torch.cat(pieces, dim=1)
        merged_mask = torch.cat(mask_pieces, dim=1).unsqueeze(-1)
        pooled = (merged * merged_mask).sum(dim=1) / merged_mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class ProjectionHead(nn.Module):
    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(in_dim), int(out_dim), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@dataclass(frozen=True)
class PromptLearningConfig:
    vocab_size: int
    pad_id: int
    max_text_length: int
    image_size: int
    prompt_length: int = 4
    embed_dim: int = 32
    vision_width: int = 32
    text_width: int = 32
    init_temperature: float = 0.07


class ToyPromptLearningVLM(nn.Module):
    def __init__(self, cfg: PromptLearningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(width=int(cfg.vision_width))
        self.text_encoder = FrozenTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            width=int(cfg.text_width),
        )
        self.image_projection = ProjectionHead(in_dim=int(cfg.vision_width), out_dim=int(cfg.embed_dim))
        self.text_projection = ProjectionHead(in_dim=int(cfg.text_width), out_dim=int(cfg.embed_dim))
        self.soft_prompt = nn.Parameter(torch.randn(int(cfg.prompt_length), int(cfg.text_width)) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / float(cfg.init_temperature))))
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        modules = [
            self.vision_encoder,
            self.text_encoder,
            self.image_projection,
            self.text_projection,
        ]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = batch["input_ids"]
        if int(input_ids.shape[1]) != int(self.cfg.max_text_length):
            raise ValueError(
                "input_ids length does not match model config: "
                f"{int(input_ids.shape[1])} != {int(self.cfg.max_text_length)}"
            )

        prompt_embed = self.soft_prompt.unsqueeze(0).expand(int(input_ids.shape[0]), -1, -1)
        image_features = self.vision_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids,
            batch["attention_mask"],
            prompt_embed=prompt_embed,
        )

        image_embed = F.normalize(self.image_projection(image_features), dim=-1)
        text_embed = F.normalize(self.text_projection(text_features), dim=-1)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * image_embed @ text_embed.transpose(0, 1)
        logits_per_text = logits_per_image.transpose(0, 1)
        return {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "prompt_embed": prompt_embed,
        }


def clip_contrastive_loss(
    logits_per_image: torch.Tensor, logits_per_text: torch.Tensor
) -> torch.Tensor:
    target = torch.arange(int(logits_per_image.shape[0]), device=logits_per_image.device)
    image_loss = F.cross_entropy(logits_per_image, target)
    text_loss = F.cross_entropy(logits_per_text, target)
    return 0.5 * (image_loss + text_loss)


@torch.no_grad()
def retrieval_accuracy(
    logits_per_image: torch.Tensor, logits_per_text: torch.Tensor
) -> tuple[float, float]:
    target = torch.arange(int(logits_per_image.shape[0]), device=logits_per_image.device)
    image_to_text = (logits_per_image.argmax(dim=-1) == target).to(torch.float32).mean()
    text_to_image = (logits_per_text.argmax(dim=-1) == target).to(torch.float32).mean()
    return float(image_to_text.item()), float(text_to_image.item())


__all__ = [
    "FrozenTextEncoder",
    "PromptLearningConfig",
    "ProjectionHead",
    "ToyPromptLearningVLM",
    "VisionEncoder",
    "clip_contrastive_loss",
    "retrieval_accuracy",
]
