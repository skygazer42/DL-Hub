from __future__ import annotations

import torch
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "tac_sum_tiny": {"width": 24, "depth": 2},
    "tac_sum_small": {"width": 32, "depth": 3},
    "tac_sum_base": {"width": 48, "depth": 4},
}


class TACSUMVideoSummarizer(nn.Module):
    """TAC-SUM-style training-free temporal-aware clustering summarizer (toy).

    The scoring logic is intentionally simple and deterministic:
    - encode frames
    - build a few temporal prototypes from evenly spaced anchors
    - score each frame by prototype affinity + temporal novelty
    """

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)  # (B,T,D)
        b, t, d = feat.shape
        norm_feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        num_proto = max(2, min(4, int(t)))
        proto_idx = torch.linspace(0, int(t) - 1, steps=num_proto, device=feat.device).round().long()
        prototypes = norm_feat[:, proto_idx]  # (B,P,D)
        affinity = torch.einsum("btd,bpd->btp", norm_feat, prototypes).amax(dim=-1)

        novelty = torch.zeros(int(b), int(t), device=feat.device, dtype=feat.dtype)
        if int(t) > 1:
            novelty[:, 1:] = (norm_feat[:, 1:] - norm_feat[:, :-1]).pow(2).mean(dim=-1).sqrt()
        time_pos = torch.linspace(0.0, 1.0, steps=int(t), device=feat.device).view(1, int(t))
        center_bias = 1.0 - (time_pos - 0.5).abs() * 1.2

        raw = 0.55 * affinity + 0.30 * novelty + 0.15 * center_bias
        scores = torch.sigmoid(raw)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "prototype_affinity": affinity,
        }


def build_tac_sum_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "tac_sum_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown TAC-SUM variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return TACSUMVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_tac_sum_video_summarizer(in_channels=3, variant="tac_sum_tiny", width_mult=0.5)
    out = m(x)
    print("tac_sum_tiny", tuple(out["scores"].shape), tuple(out["prototype_affinity"].shape))
    loss = out["scores"].mean() + out["prototype_affinity"].mean()
    loss.backward()
    print("ok")

