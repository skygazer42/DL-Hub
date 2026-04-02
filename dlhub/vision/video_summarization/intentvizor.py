from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "intentvizor_tiny": {"width": 24, "depth": 2},
    "intentvizor_small": {"width": 32, "depth": 3},
    "intentvizor_base": {"width": 48, "depth": 4},
}


def _prepare_query_state(
    query: torch.Tensor | None,
    *,
    batch: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    prompt: torch.Tensor,
) -> torch.Tensor:
    if query is None:
        return prompt.mean(dim=0, keepdim=True).expand(int(batch), -1)

    q = query.to(device=device, dtype=dtype)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    elif q.ndim == 3:
        q = q.mean(dim=1)
    elif q.ndim != 2:
        raise ValueError(f"query must have shape (D), (B,D) or (B,Q,D), got {tuple(q.shape)}")

    if int(q.shape[0]) == 1 and int(batch) > 1:
        q = q.expand(int(batch), -1)
    elif int(q.shape[0]) != int(batch):
        raise ValueError(f"query batch {int(q.shape[0])} does not match video batch {int(batch)}")

    cur_dim = int(q.shape[-1])
    if cur_dim < int(dim):
        pad = torch.zeros(int(batch), int(dim) - cur_dim, device=device, dtype=dtype)
        q = torch.cat([q, pad], dim=-1)
    elif cur_dim > int(dim):
        q = q[..., : int(dim)]
    return q


class IntentVizorVideoSummarizer(nn.Module):
    """Intent-guided summarizer with lightweight multi-granularity ego-graph mixing."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.radii = (1, 2, 4)
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        hidden = max(32, dim // 2)
        num_intents = max(4, int(depth) + 2)

        self.intent_prompt = nn.Parameter(torch.randn(max(2, int(depth)), dim) * 0.02)
        self.intent_basis = nn.Parameter(torch.randn(num_intents, dim) * 0.02)
        self.intent_proj = nn.Linear(dim, dim)
        self.ego_projs = nn.ModuleList(nn.Linear(dim, dim) for _ in self.radii)
        self.head = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def _ego_graph_mix(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = feat.shape
        norm = F.normalize(feat, dim=-1)
        sim = torch.matmul(norm, norm.transpose(1, 2))
        pos = torch.arange(int(t), device=feat.device)
        dist = (pos.view(1, int(t), 1) - pos.view(1, 1, int(t))).abs()

        mixed: list[torch.Tensor] = []
        maps: list[torch.Tensor] = []
        for radius, proj in zip(self.radii, self.ego_projs, strict=True):
            mask = dist <= int(radius)
            attn = torch.softmax(sim.masked_fill(~mask, -1e4), dim=-1)
            mixed.append(torch.matmul(attn, proj(feat)))
            maps.append(attn.unsqueeze(1))
        return torch.stack(mixed, dim=0).mean(dim=0), torch.cat(maps, dim=1)

    def forward(
        self,
        video: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        query_state = _prepare_query_state(
            query,
            batch=int(b),
            dim=int(d),
            device=feat.device,
            dtype=feat.dtype,
            prompt=self.intent_prompt.to(device=feat.device, dtype=feat.dtype),
        )
        basis = self.intent_basis.to(device=feat.device, dtype=feat.dtype)
        intent_logits = torch.matmul(self.intent_proj(query_state), basis.transpose(0, 1)) / math.sqrt(
            max(1, int(d))
        )
        intent_probs = torch.softmax(intent_logits, dim=-1)
        intent_state = torch.matmul(intent_probs, basis)

        graph_feat, graph_maps = self._ego_graph_mix(feat)
        intent_expand = intent_state.unsqueeze(1).expand(-1, int(t), -1)
        fused = torch.cat([feat, graph_feat, feat * torch.tanh(intent_expand)], dim=-1)
        raw_scores = self.head(fused).squeeze(-1)
        intent_alignment = F.normalize(feat, dim=-1).mul(F.normalize(intent_expand, dim=-1)).sum(dim=-1)
        scores = torch.sigmoid(raw_scores + 0.35 * intent_alignment)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "intent_probs": intent_probs,
            "ego_graph_maps": graph_maps,
        }


def build_intentvizor_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "intentvizor_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown IntentVizor variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return IntentVizorVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_intentvizor_video_summarizer(
        in_channels=3,
        variant="intentvizor_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("intentvizor_tiny", tuple(out["scores"].shape), tuple(out["intent_probs"].shape))
    loss = out["scores"].mean() + out["intent_probs"].mean()
    loss.backward()
    print("ok")
