from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SAMPrompt:
    point: tuple[int, int] | None = None
    box: tuple[int, int, int, int] | None = None
    mask: torch.Tensor | None = None
    text: str = ""


@dataclass(frozen=True)
class SAMConfig:
    image_size: int
    patch_size: int = 16
    embed_dim: int = 256
    num_heads: int = 8
    num_prompt_masks: int = 3
    iou_head_hidden_dim: int = 128


@dataclass(frozen=True)
class SAMDataEngine:
    num_masks: int = 1_100_000_000
    num_images: int = 11_000_000
    average_masks_per_image: int = 100
    privacy_respecting: bool = True
    licensed_images: bool = True


class SAMPromptEncoder(nn.Module):
    supports_text_prompts = True

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(2, int(embed_dim)),
            nn.GELU(),
            nn.Linear(int(embed_dim), int(embed_dim)),
        )
        self.box_mlp = nn.Sequential(
            nn.Linear(4, int(embed_dim)),
            nn.GELU(),
            nn.Linear(int(embed_dim), int(embed_dim)),
        )
        self.mask_pool = nn.AdaptiveAvgPool2d(1)

    def forward(
        self,
        prompts: tuple[SAMPrompt, ...],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not prompts:
            prompts = tuple(SAMPrompt() for _ in range(batch_size))
        if len(prompts) == 1 and batch_size > 1:
            prompts = prompts * batch_size
        if len(prompts) != batch_size:
            raise ValueError(f"expected {batch_size} prompts, got {len(prompts)}")

        embeddings: list[torch.Tensor] = []
        for prompt in prompts:
            embedding = torch.zeros(self.embed_dim, device=device)
            if prompt.point is not None:
                point = torch.tensor(prompt.point, device=device, dtype=torch.float32)
                embedding = embedding + self.point_mlp(point)
            if prompt.box is not None:
                box = torch.tensor(prompt.box, device=device, dtype=torch.float32)
                embedding = embedding + self.box_mlp(box)
            if prompt.mask is not None:
                pooled = self.mask_pool(prompt.mask.to(device=device, dtype=torch.float32).unsqueeze(0))
                embedding = embedding + pooled.flatten()
            if prompt.text:
                scale = torch.tensor([len(prompt.text)], device=device, dtype=torch.float32)
                embedding = embedding + scale.repeat(self.embed_dim) / max(1, self.embed_dim)
            embeddings.append(embedding)
        return torch.stack(embeddings, dim=0)


class SAMMaskDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_prompt_masks: int, iou_head_hidden_dim: int) -> None:
        super().__init__()
        self.mask_head = nn.Conv2d(int(embed_dim), int(num_prompt_masks), kernel_size=1)
        self.iou_head = nn.Sequential(
            nn.Linear(int(embed_dim), int(iou_head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(iou_head_hidden_dim), int(num_prompt_masks)),
        )

    def forward(self, image_embedding: torch.Tensor, prompt_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = image_embedding + prompt_embedding.unsqueeze(-1).unsqueeze(-1)
        mask_logits = self.mask_head(fused)
        iou_scores = self.iou_head(fused.mean(dim=(2, 3)))
        return {"mask_logits": mask_logits, "iou_scores": iou_scores}


class SAMAutomaticMaskGenerator:
    def __init__(self, points_per_side: int = 32) -> None:
        self.points_per_side = int(points_per_side)

    def build_grid_prompts(self, image_size: int) -> tuple[SAMPrompt, ...]:
        step = max(1, int(image_size) // max(1, self.points_per_side))
        offset = step // 2
        prompts = []
        for y in range(self.points_per_side):
            for x in range(self.points_per_side):
                prompts.append(
                    SAMPrompt(
                        point=(
                            min(int(image_size) - 1, offset + x * step),
                            min(int(image_size) - 1, offset + y * step),
                        )
                    )
                )
        return tuple(prompts)


class HQSAMModel(nn.Module):
    ambiguity_aware = True

    def __init__(self, config: SAMConfig) -> None:
        super().__init__()
        self.config = config
        self.data_engine = SAMDataEngine()
        self.image_encoder = nn.Conv2d(
            3,
            int(config.embed_dim),
            kernel_size=int(config.patch_size),
            stride=int(config.patch_size),
        )
        self.prompt_encoder = SAMPromptEncoder(int(config.embed_dim))
        self.mask_decoder = SAMMaskDecoder(
            int(config.embed_dim),
            int(config.num_prompt_masks),
            int(config.iou_head_hidden_dim),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image)

    def forward(
        self,
        *,
        image: torch.Tensor,
        prompts: tuple[SAMPrompt, ...] = (),
    ) -> dict[str, torch.Tensor]:
        image_embedding = self.encode_image(image)
        prompt_embedding = self.prompt_encoder(
            prompts,
            batch_size=image.shape[0],
            device=image.device,
        )
        output = self.mask_decoder(image_embedding, prompt_embedding)
        output["mask_logits"] = F.interpolate(
            output["mask_logits"],
            size=(int(self.config.image_size), int(self.config.image_size)),
            mode="bilinear",
            align_corners=False,
        )
        return output


__all__ = [
    "SAMAutomaticMaskGenerator",
    "SAMConfig",
    "SAMDataEngine",
    "SAMMaskDecoder",
    "SAMPrompt",
    "SAMPromptEncoder",
    "HQSAMModel",
]

