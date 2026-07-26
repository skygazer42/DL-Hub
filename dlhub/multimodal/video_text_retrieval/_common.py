from __future__ import annotations

import torch
from torch import nn


def check_video(video: torch.Tensor) -> torch.Tensor:
    video = video.to(torch.float32)
    if video.ndim != 5:
        raise ValueError(f"Expected input shape (B, T, C, H, W), got {tuple(video.shape)}")
    return video


class TinyVideoTextRetriever(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        layers: list[nn.Module] = [
            nn.Conv2d(int(in_channels), int(width), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend(
                [
                    nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        self.frame_encoder = nn.Sequential(*layers)
        self.frame_pool = nn.AdaptiveAvgPool2d(1)
        self.temporal = nn.GRU(int(width), int(width), batch_first=True)
        self.text_encoder = nn.Sequential(
            nn.Linear(32, int(width)),
            nn.ReLU(inplace=True),
            nn.Linear(int(width), int(width)),
        )
        self.fusion = nn.Linear(int(width), int(width))

    def forward(
        self, video: torch.Tensor, text: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        clip = check_video(video)
        batch, frames, _, _, _ = clip.shape
        clip_2d = clip.reshape(batch * frames, *clip.shape[2:])
        encoded = self.frame_encoder(clip_2d)
        frame_tokens = self.frame_pool(encoded).flatten(1).reshape(batch, frames, -1)
        video_embedding, _ = self.temporal(frame_tokens)
        video_embedding = video_embedding[:, -1]

        text = (
            torch.zeros(batch, 32, dtype=video_embedding.dtype, device=video_embedding.device)
            if text is None
            else text.to(torch.float32)
        )
        text_embedding = self.text_encoder(text)

        if self.mode == "clip4clip":
            fused_video = video_embedding
        elif self.mode == "xpool":
            fused_video = 0.7 * video_embedding + 0.3 * frame_tokens.mean(dim=1)
        elif self.mode == "frozen":
            fused_video = torch.tanh(video_embedding)
        elif self.mode == "dual":
            fused_video = self.fusion(video_embedding)
        elif self.mode == "cross":
            fused_video = self.fusion(video_embedding + text_embedding)
        elif self.mode == "temporal":
            fused_video = video_embedding + frame_tokens.mean(dim=1)
        elif self.mode == "transformer":
            fused_video = video_embedding * torch.sigmoid(self.fusion(text_embedding))
        elif self.mode == "retrieval_aug":
            fused_video = self.fusion(video_embedding + 0.5 * text_embedding)
        elif self.mode == "prompt":
            fused_video = self.fusion(video_embedding) + 0.1 * text_embedding
        elif self.mode == "mamba":
            fused_video = video_embedding + torch.roll(video_embedding, shifts=1, dims=-1)
        else:
            fused_video = video_embedding

        similarity = (fused_video * text_embedding).sum(dim=1, keepdim=True)
        return {
            "video_embedding": fused_video,
            "text_embedding": text_embedding,
            "similarity": similarity,
        }


def build_toy_vtr(
    *,
    family: str,
    mode: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    cfg = variants[str(variant)]
    width = max(16, int(int(cfg["width"]) * float(width_mult)))
    return TinyVideoTextRetriever(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
    )


def smoke_test_vtr(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 4, 3, 32, 32), torch.randn(2, 32))
    print(variant, tuple(out["video_embedding"].shape), tuple(out["similarity"].shape))
