from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "sem_reward_rl_tiny": {"width": 24, "depth": 2},
    "sem_reward_rl_small": {"width": 32, "depth": 3},
    "sem_reward_rl_base": {"width": 48, "depth": 4},
}


class SemRewardRLVideoSummarizer(nn.Module):
    """Semantic-reward RL-style summarizer with proxy reward decomposition."""

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
        self.policy = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, _ = feat.shape
        raw_policy = self.policy(feat).squeeze(-1)
        base_scores = torch.sigmoid(raw_policy)

        summary_token = torch.einsum("bt,btd->bd", torch.softmax(base_scores, dim=-1), feat)
        full_token = feat.mean(dim=1)
        semantic_reward = F.cosine_similarity(summary_token, full_token, dim=-1).unsqueeze(1).expand(-1, int(t))

        diversity_reward = torch.zeros(int(b), int(t), device=feat.device, dtype=feat.dtype)
        if int(t) > 1:
            diversity_reward[:, 1:] = (
                F.normalize(feat[:, 1:], dim=-1) - F.normalize(feat[:, :-1], dim=-1)
            ).pow(2).mean(dim=-1).sqrt()

        represent_reward = torch.matmul(
            F.normalize(feat, dim=-1),
            F.normalize(full_token, dim=-1).unsqueeze(-1),
        ).squeeze(-1)

        reward_logits = raw_policy + 0.35 * semantic_reward + 0.20 * diversity_reward + 0.20 * represent_reward
        scores = torch.sigmoid(reward_logits)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "semantic_reward": semantic_reward,
            "diversity_reward": diversity_reward,
        }


def build_sem_reward_rl_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "sem_reward_rl_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Sem-Reward-RL variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return SemRewardRLVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_sem_reward_rl_video_summarizer(
        in_channels=3,
        variant="sem_reward_rl_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("sem_reward_rl_tiny", tuple(out["scores"].shape), tuple(out["semantic_reward"].shape))
    loss = out["scores"].mean() + out["diversity_reward"].mean()
    loss.backward()
    print("ok")
