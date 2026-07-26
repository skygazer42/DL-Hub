from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "personalized_ranker_tiny": {"width": 24, "depth": 2},
    "personalized_ranker_small": {"width": 32, "depth": 3},
    "personalized_ranker_base": {"width": 48, "depth": 4},
}


def _prepare_preference(
    preference: torch.Tensor | None,
    *,
    batch: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    bank: torch.Tensor,
) -> torch.Tensor:
    if preference is None:
        return bank.mean(dim=0, keepdim=True).expand(int(batch), -1)

    pref = preference.to(device=device, dtype=dtype)
    if pref.ndim == 1:
        pref = pref.unsqueeze(0)
    elif pref.ndim != 2:
        raise ValueError(f"preference must have shape (D) or (B,D), got {tuple(pref.shape)}")

    if int(pref.shape[0]) == 1 and int(batch) > 1:
        pref = pref.expand(int(batch), -1)
    elif int(pref.shape[0]) != int(batch):
        raise ValueError(
            f"preference batch {int(pref.shape[0])} does not match video batch {int(batch)}"
        )

    cur_dim = int(pref.shape[-1])
    if cur_dim < int(dim):
        pad = torch.zeros(int(batch), int(dim) - cur_dim, device=device, dtype=dtype)
        pref = torch.cat([pref, pad], dim=-1)
    elif cur_dim > int(dim):
        pref = pref[..., : int(dim)]
    return pref


class PersonalizedRankerVideoSummarizer(nn.Module):
    """Multiple pairwise ranking summarizer with optional preference conditioning."""

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
        self.num_rankers = max(3, int(depth) + 1)
        self.preference_bank = nn.Parameter(torch.randn(self.num_rankers, dim) * 0.02)
        self.local_rankers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dim * 2, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
                nn.Linear(hidden, 1),
            )
            for _ in range(self.num_rankers)
        )
        self.global_head = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        video: torch.Tensor,
        preference: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        bank = self.preference_bank.to(device=feat.device, dtype=feat.dtype)
        pref = _prepare_preference(
            preference,
            batch=int(b),
            dim=int(d),
            device=feat.device,
            dtype=feat.dtype,
            bank=bank,
        )
        pref_expand = pref.unsqueeze(1).expand(-1, int(t), -1)

        local_scores = []
        for idx, head in enumerate(self.local_rankers):
            bank_expand = bank[idx].view(1, 1, int(d)).expand(int(b), int(t), int(d))
            local_scores.append(
                head(torch.cat([feat, feat * torch.tanh(bank_expand)], dim=-1)).squeeze(-1)
            )
        local_rankings = torch.stack(local_scores, dim=1)  # (B,R,T)

        preference_logits = torch.matmul(
            F.normalize(pref, dim=-1), F.normalize(bank, dim=-1).transpose(0, 1)
        )
        preference_weights = torch.softmax(preference_logits, dim=-1)
        personalized_logits = torch.einsum("br,brt->bt", preference_weights, local_rankings)
        global_logits = self.global_head(
            torch.cat([feat, feat * torch.tanh(pref_expand)], dim=-1)
        ).squeeze(-1)
        scores = torch.sigmoid(torch.maximum(global_logits, personalized_logits))
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "local_rankings": local_rankings,
            "preference_weights": preference_weights,
        }


def build_personalized_ranker_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "personalized_ranker_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Personalized-Ranker variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return PersonalizedRankerVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_personalized_ranker_video_summarizer(
        in_channels=3,
        variant="personalized_ranker_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print(
        "personalized_ranker_tiny",
        tuple(out["scores"].shape),
        tuple(out["local_rankings"].shape),
    )
    loss = out["scores"].mean() + out["preference_weights"].mean()
    loss.backward()
    print("ok")
