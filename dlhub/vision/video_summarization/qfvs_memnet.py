from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "qfvs_memnet_tiny": {"width": 24, "depth": 2},
    "qfvs_memnet_small": {"width": 32, "depth": 3},
    "qfvs_memnet_base": {"width": 48, "depth": 4},
}


def _prepare_query(
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


class QFVSMemNetVideoSummarizer(nn.Module):
    """Query-focused video summarizer with a tiny learned memory network."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        hidden = max(32, dim // 2)
        mem_slots = max(3, int(depth) + 1)
        prompt_tokens = max(2, int(depth))

        self.query_prompt = nn.Parameter(torch.randn(prompt_tokens, dim) * 0.02)
        self.memory_bank = nn.Parameter(torch.randn(mem_slots, dim) * 0.02)
        self.frame_proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(dim, dim)
        self.mem_proj = nn.Linear(dim, dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        video: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        query_vec = _prepare_query(
            query,
            batch=int(b),
            dim=int(d),
            device=feat.device,
            dtype=feat.dtype,
            prompt=self.query_prompt.to(device=feat.device, dtype=feat.dtype),
        )

        memory = self.memory_bank.to(device=feat.device, dtype=feat.dtype).unsqueeze(0).expand(int(b), -1, -1)
        q = self.query_proj(query_vec)
        mem_key = self.mem_proj(memory)
        mem_attn = torch.softmax(
            torch.einsum("bd,bmd->bm", q, mem_key) / math.sqrt(max(1, int(d))),
            dim=-1,
        )
        mem_ctx = torch.einsum("bm,bmd->bd", mem_attn, memory)

        frame_key = self.frame_proj(feat)
        frame_attn = torch.softmax(
            torch.einsum("btd,bd->bt", frame_key, q + mem_ctx) / math.sqrt(max(1, int(d))),
            dim=-1,
        )
        readout = torch.einsum("bt,btd->bd", frame_attn, feat)
        state = 0.5 * q + 0.3 * mem_ctx + 0.2 * readout

        state_expand = state.unsqueeze(1).expand(-1, int(t), -1)
        mem_expand = mem_ctx.unsqueeze(1).expand(-1, int(t), -1)
        fused = torch.cat([feat, feat * torch.tanh(state_expand), mem_expand], dim=-1)
        raw_scores = self.head(fused).squeeze(-1) + frame_attn
        scores = torch.sigmoid(raw_scores)
        summary_mask = scores_to_mask(scores)
        query_alignment = F.normalize(frame_key, dim=-1).mul(
            F.normalize(state_expand, dim=-1)
        ).sum(dim=-1)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "query_alignment": query_alignment,
            "memory_attention": mem_attn,
        }


def build_qfvs_memnet_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "qfvs_memnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown QFVS-MemNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return QFVSMemNetVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_qfvs_memnet_video_summarizer(
        in_channels=3,
        variant="qfvs_memnet_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("qfvs_memnet_tiny", tuple(out["scores"].shape), tuple(out["memory_attention"].shape))
    loss = out["scores"].mean() + out["memory_attention"].mean()
    loss.backward()
    print("ok")
