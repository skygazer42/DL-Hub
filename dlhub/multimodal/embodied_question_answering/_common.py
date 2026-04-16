from __future__ import annotations

import torch
from torch import nn


def check_observations(observations: torch.Tensor) -> torch.Tensor:
    observations = observations.to(torch.float32)
    if observations.ndim != 5:
        raise ValueError(f"Expected input shape (B, T, C, H, W), got {tuple(observations.shape)}")
    return observations


class TinyEmbodiedQAModel(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
        question_dim: int,
        num_answers: int,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.question_dim = int(question_dim)
        self.num_answers = int(num_answers)

        layers: list[nn.Module] = [
            nn.Conv2d(int(in_channels), int(width), kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend(
                [
                    nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
        self.observation_encoder = nn.Sequential(*layers)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.temporal = nn.GRU(int(width), int(width), batch_first=True)
        self.question_encoder = nn.Sequential(
            nn.Linear(int(question_dim), int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
        )
        self.map_adapter = nn.Linear(int(width), int(width))
        self.memory_cell = nn.GRUCell(int(width), int(width))
        self.answer_head = nn.Sequential(
            nn.Linear(int(width), int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(num_answers)),
        )

    def forward(
        self,
        observations: torch.Tensor,
        question: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        clip = check_observations(observations)
        batch, frames, _, _, _ = clip.shape
        clip_2d = clip.reshape(batch * frames, *clip.shape[2:])
        encoded = self.observation_encoder(clip_2d)
        frame_tokens = self.spatial_pool(encoded).flatten(1).reshape(batch, frames, -1)
        trajectory, _ = self.temporal(frame_tokens)
        route_state = trajectory[:, -1]

        if question is None:
            question = torch.zeros(batch, self.question_dim, dtype=route_state.dtype, device=route_state.device)
        else:
            question = question.to(torch.float32)
        question_state = self.question_encoder(question)

        if self.mode == "navqa":
            fused = route_state + 0.25 * frame_tokens.mean(dim=1)
        elif self.mode == "memory":
            memory = torch.zeros_like(route_state)
            for step in range(frames):
                memory = self.memory_cell(frame_tokens[:, step], memory)
            fused = memory + 0.5 * question_state
        elif self.mode == "objectnav":
            object_hint = encoded.flatten(2).amax(dim=2).reshape(batch, frames, -1).mean(dim=1)
            fused = route_state + object_hint + 0.25 * question_state
        elif self.mode == "mapqa":
            fused = self.map_adapter(frame_tokens.mean(dim=1)) + question_state
        elif self.mode == "speaker":
            fused = route_state + torch.tanh(question_state)
        elif self.mode == "transformer":
            fused = route_state * torch.sigmoid(question_state)
        elif self.mode == "grounded":
            grounded = frame_tokens.mean(dim=1) * torch.sigmoid(question_state)
            fused = route_state + grounded
        elif self.mode == "retrieval":
            similarity = torch.softmax(frame_tokens @ question_state.unsqueeze(-1), dim=1)
            retrieved = (frame_tokens * similarity).sum(dim=1)
            fused = route_state + retrieved
        elif self.mode == "prompt":
            fused = route_state + question_state + 0.1 * self.map_adapter(question_state)
        elif self.mode == "mamba":
            fused = route_state + torch.roll(route_state, shifts=1, dims=-1) + 0.25 * question_state
        else:
            fused = route_state + question_state

        logits = self.answer_head(fused)
        return {
            "logits": logits,
            "answer_embedding": fused,
            "question_embedding": question_state,
        }


def build_toy_eqa(
    *,
    family: str,
    mode: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    question_dim: int = 32,
    num_answers: int = 8,
) -> nn.Module:
    cfg = variants[str(variant)]
    width = max(16, int(int(cfg["width"]) * float(width_mult)))
    return TinyEmbodiedQAModel(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        question_dim=int(question_dim),
        num_answers=int(num_answers),
    )


def smoke_test_eqa(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 4, 3, 32, 32), torch.randn(2, 32))
    print(variant, tuple(out["logits"].shape), tuple(out["answer_embedding"].shape))
