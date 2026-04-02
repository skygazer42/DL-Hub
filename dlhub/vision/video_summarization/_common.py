from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_video_input(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 5:
        raise ValueError(f"Expected video input shape (B, T, C, H, W), got {tuple(x.shape)}")
    return x


def scores_to_mask(scores: torch.Tensor, *, keep_ratio: float = 0.3) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (B, T), got {tuple(scores.shape)}")
    b, t = scores.shape
    keep = max(1, min(int(t), int(round(float(keep_ratio) * int(t)))))
    idx = scores.topk(keep, dim=1).indices
    mask = torch.zeros_like(scores)
    mask.scatter_(1, idx, 1.0)
    return mask


class TinyFrameEncoder(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c_in = int(in_channels)
        w = int(width)
        d = max(1, int(depth))
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w < 8:
            raise ValueError("width must be >= 8")

        layers: list[nn.Module] = [
            nn.Conv2d(c_in, w, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        cur = w
        for i in range(d):
            out = cur if i == 0 else cur * 2
            layers.extend(
                [
                    nn.Conv2d(cur, out, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            if i < d - 1:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            cur = out

        self.net = nn.Sequential(*layers)
        self.out_dim = int(cur)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video = check_video_input(video)
        b, t, c, h, w = video.shape
        frames = video.reshape(int(b) * int(t), int(c), int(h), int(w))
        feat = self.net(frames)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return feat.view(int(b), int(t), -1)


class TemporalGRUScorer(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int, layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(hidden_dim)
        n = max(1, int(layers))
        self.rnn = nn.GRU(
            input_size=d,
            hidden_size=h,
            num_layers=n,
            batch_first=True,
            dropout=float(dropout) if n > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(h, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 3:
            raise ValueError(f"feat must have shape (B, T, D), got {tuple(feat.shape)}")
        seq, _ = self.rnn(feat)
        return self.head(seq).squeeze(-1)


class TemporalAttentionScorer(nn.Module):
    def __init__(self, *, dim: int, heads: int = 4, depth: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(heads)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=h,
            dim_feedforward=max(64, d * 4),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(depth)))
        self.head = nn.Linear(d, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 3:
            raise ValueError(f"feat must have shape (B, T, D), got {tuple(feat.shape)}")
        out = self.encoder(feat)
        return self.head(out).squeeze(-1)


class SegmentPooler(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(h, 1),
        )

    def forward(self, feat: torch.Tensor, *, windows: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        if feat.ndim != 3:
            raise ValueError(f"feat must have shape (B, T, D), got {tuple(feat.shape)}")
        b, t, _ = feat.shape
        frame_scores = torch.zeros(int(b), int(t), device=feat.device, dtype=feat.dtype)
        counts = torch.zeros_like(frame_scores)
        seg_scores: list[torch.Tensor] = []
        for win in windows:
            w = max(1, int(win))
            if w > int(t):
                continue
            chunks = []
            for start in range(0, int(t) - w + 1):
                seg = feat[:, start : start + w].mean(dim=1)
                score = self.proj(seg).squeeze(-1)
                frame_scores[:, start : start + w] = frame_scores[:, start : start + w] + score.unsqueeze(1)
                counts[:, start : start + w] = counts[:, start : start + w] + 1.0
                chunks.append(score.unsqueeze(1))
            if chunks:
                seg_scores.append(torch.cat(chunks, dim=1))
        frame_scores = frame_scores / counts.clamp_min(1.0)
        if seg_scores:
            max_len = max(int(x.shape[1]) for x in seg_scores)
            padded = []
            for x in seg_scores:
                if int(x.shape[1]) < max_len:
                    pad = torch.zeros(int(b), max_len - int(x.shape[1]), device=x.device, dtype=x.dtype)
                    x = torch.cat([x, pad], dim=1)
                padded.append(x.unsqueeze(1))
            seg_tensor = torch.cat(padded, dim=1)
        else:
            seg_tensor = torch.zeros(int(b), 0, 0, device=feat.device, dtype=feat.dtype)
        return frame_scores, seg_tensor


def _default_variants(prefix: str) -> dict[str, dict[str, int]]:
    p = str(prefix).strip()
    return {
        f"{p}_tiny": {"width": 24, "depth": 2},
        f"{p}_small": {"width": 32, "depth": 3},
        f"{p}_base": {"width": 48, "depth": 4},
    }
