from __future__ import annotations

import torch
from torch import nn


def check_audio(audio: torch.Tensor) -> torch.Tensor:
    audio = audio.to(torch.float32)
    if audio.ndim != 3:
        raise ValueError(f"Expected input shape (B, C, T), got {tuple(audio.shape)}")
    return audio


class TinyAudioTextUnderstandingModel(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)

        layers: list[nn.Module] = [
            nn.Conv1d(int(in_channels), int(width), kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend(
                [
                    nn.Conv1d(int(width), int(width), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        self.audio_encoder = nn.Sequential(*layers)
        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        self.audio_gate = nn.Linear(int(width), int(width))
        self.text_encoder = nn.Sequential(
            nn.Linear(32, int(width)),
            nn.ReLU(inplace=True),
            nn.Linear(int(width), int(width)),
        )
        self.fusion = nn.Linear(int(width), int(width))
        self.classifier = nn.Linear(int(width), 4)

    def forward(self, audio: torch.Tensor, text: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        clip = check_audio(audio)
        audio_embedding = self.audio_pool(self.audio_encoder(clip)).flatten(1)

        batch = clip.shape[0]
        text = (
            torch.zeros(batch, 32, dtype=audio_embedding.dtype, device=audio_embedding.device)
            if text is None
            else text.to(torch.float32)
        )
        text_embedding = self.text_encoder(text)

        if self.mode == "audio_bert":
            fused = self.fusion(audio_embedding + text_embedding)
        elif self.mode == "wav2text":
            fused = self.fusion(0.8 * audio_embedding + 0.2 * text_embedding)
        elif self.mode == "contrastive":
            fused = audio_embedding - text_embedding
        elif self.mode == "event":
            fused = self.fusion(audio_embedding + torch.tanh(text_embedding))
        elif self.mode == "speech":
            fused = self.fusion(audio_embedding * torch.sigmoid(text_embedding))
        elif self.mode == "transformer":
            fused = self.fusion(audio_embedding + self.audio_gate(text_embedding))
        elif self.mode == "retrieval":
            fused = self.fusion(0.5 * audio_embedding + text_embedding)
        elif self.mode == "diffusion":
            fused = self.fusion(audio_embedding + 0.1 * torch.randn_like(audio_embedding))
        elif self.mode == "prompt":
            fused = self.fusion(audio_embedding) + 0.1 * text_embedding
        elif self.mode == "mamba":
            fused = audio_embedding + torch.roll(audio_embedding, shifts=1, dims=-1)
        else:
            fused = audio_embedding

        alignment = (fused * text_embedding).sum(dim=1, keepdim=True)
        return {
            "audio_embedding": fused,
            "text_embedding": text_embedding,
            "alignment": alignment,
            "logits": self.classifier(fused),
        }


def build_toy_atu(
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
    return TinyAudioTextUnderstandingModel(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
    )


def smoke_test_atu(builder, variant: str) -> None:
    model = builder(in_channels=1, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 1, 128), torch.randn(2, 32))
    print(variant, tuple(out["audio_embedding"].shape), tuple(out["alignment"].shape))
