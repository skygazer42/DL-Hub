from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from dlhub.pointcloud.detection3d._common import (
    EdgeConv,
    PointNetEncoder,
    TinyTransformerEncoder,
    sinusoidal_positional_encoding,
)
from dlhub.pointcloud.ops import farthest_point_sample, index_points


def check_points(points: torch.Tensor) -> torch.Tensor:
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points).__name__}")
    if points.ndim != 3:
        raise ValueError(f"points must have shape (B, N, C), got {tuple(points.shape)}")
    if points.shape[-1] < 3:
        raise ValueError(f"points last dim must be >=3 (xyz), got C={points.shape[-1]}")
    return points.to(torch.float32)


def _repeat_or_sample(points: torch.Tensor, num_points: int) -> torch.Tensor:
    num_points = int(num_points)
    if num_points <= 0:
        raise ValueError("num_points must be > 0")
    b, n, _ = points.shape
    if n == 0:
        raise ValueError("points must contain at least one point")
    if n >= num_points:
        idx = farthest_point_sample(points[..., :3], num_points)
        return index_points(points, idx)

    reps = math.ceil(num_points / n)
    return points.repeat(1, reps, 1)[:, :num_points, :]


def _prompt_stats(
    prompt: str | list[str] | tuple[str, ...] | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    stats = torch.zeros(batch_size, 4, device=device, dtype=dtype)
    if prompt is None:
        return stats

    if isinstance(prompt, str):
        prompts = [prompt] * batch_size
    else:
        prompts = [str(item) for item in prompt]
        if len(prompts) == 1 and batch_size > 1:
            prompts = prompts * batch_size
        if len(prompts) != batch_size:
            raise ValueError(
                f"prompt batch mismatch: expected 1 or {batch_size} items, got {len(prompts)}"
            )

    rows: list[list[float]] = []
    for text in prompts:
        rows.append(
            [
                float(len(text)),
                float(text.count(" ")),
                float(sum(ch.isalpha() for ch in text)),
                float(sum(ord(ch) for ch in text) % 997),
            ]
        )

    values = torch.tensor(rows, device=device, dtype=dtype)
    scale = torch.tensor([128.0, 32.0, 128.0, 997.0], device=device, dtype=dtype)
    return values / scale


def _compatible_heads(width: int) -> int:
    for heads in (8, 4, 2, 1):
        if width % heads == 0:
            return heads
    return 1


class ToyPointCloudCompleter(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        width: int,
        depth: int,
        num_output_points: int,
        encoder_kind: str,
        decoder_kind: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.encoder_kind = str(encoder_kind)
        self.decoder_kind = str(decoder_kind)
        self.width = int(width)
        self.depth = max(1, int(depth))
        self.num_output_points = int(num_output_points)
        self.coarse_points = max(8, min(self.num_output_points, 32))
        self._uses_transformer = self.encoder_kind == "transformer" or self.decoder_kind in {
            "transformer",
            "text",
            "snowflake",
        }
        self._uses_state_space = (
            self.encoder_kind == "state_space" or self.decoder_kind == "state_space"
        )

        self.input_proj = nn.Linear(int(in_channels), self.width)
        self.pointnet = PointNetEncoder(int(in_channels), width=self.width, dropout=float(dropout))

        self.edge_layers = nn.ModuleList(
            [
                EdgeConv(self.width, self.width, k=8, dropout=float(dropout))
                for _ in range(self.depth)
            ]
        )
        self.pos_proj = (
            nn.Linear(3 * 2 * 8, self.width) if self._uses_transformer else nn.Identity()
        )
        self.transformer = (
            TinyTransformerEncoder(
                self.width,
                nhead=_compatible_heads(self.width),
                num_layers=self.depth,
                dropout=float(dropout),
            )
            if self._uses_transformer
            else nn.Identity()
        )
        self.ssm = (
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Conv1d(self.width, self.width, kernel_size=3, padding=1),
                        nn.GELU(),
                        nn.Dropout(float(dropout)),
                    )
                    for _ in range(self.depth)
                ]
            )
            if self._uses_state_space
            else nn.Identity()
        )
        self.summary_proj = nn.Linear(self.width * 2, self.width)

        self.query_embed = nn.Parameter(torch.randn(self.num_output_points, self.width) * 0.02)
        self.fold_grid = self._make_grid(self.num_output_points)
        self.grid_proj = nn.Linear(2, self.width)
        self.anchor_bank = nn.Parameter(torch.randn(max(8, self.coarse_points), 3) * 0.2)
        self.anchor_proj = nn.Linear(3, self.width)
        self.prompt_proj = nn.Linear(4, self.width)
        self.coarse_token = nn.Linear(self.width, self.coarse_points * self.width)
        self.coarse_xyz = nn.Linear(self.width, self.coarse_points * 3)
        self.query_mlp = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.width, self.width),
            nn.GELU(),
        )
        self.point_head = nn.Linear(self.width, 3)

    @staticmethod
    def _make_grid(num_points: int) -> torch.Tensor:
        side = max(2, math.ceil(math.sqrt(int(num_points))))
        values = torch.linspace(-1.0, 1.0, side)
        yy, xx = torch.meshgrid(values, values, indexing="ij")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        return grid[: int(num_points)]

    def _encode(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points[..., :3]
        if self.encoder_kind == "pointnet":
            feat = self.pointnet(points)
        elif self.encoder_kind == "edgeconv":
            feat = F.gelu(self.input_proj(points))
            for layer in self.edge_layers:
                feat = feat + layer(feat)
        elif self.encoder_kind == "transformer":
            feat = F.gelu(self.input_proj(points))
            pe = sinusoidal_positional_encoding(xyz, num_feats=8).to(feat.dtype)
            feat = self.transformer(feat + self.pos_proj(pe))
        elif self.encoder_kind == "state_space":
            feat = F.gelu(self.input_proj(points))
            feat = self.ssm(feat.transpose(1, 2)).transpose(1, 2)
        else:
            raise ValueError(f"Unsupported encoder_kind: {self.encoder_kind!r}")

        pooled = torch.cat([feat.max(dim=1).values, feat.mean(dim=1)], dim=-1)
        return self.summary_proj(pooled)

    def _query_tokens(
        self,
        summary: torch.Tensor,
        *,
        prompt: str | list[str] | tuple[str, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = summary.shape[0]
        summary_tokens = summary.unsqueeze(1)
        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1) + summary_tokens

        if self.decoder_kind in {"fold", "grid"}:
            grid = self.fold_grid.to(device=summary.device, dtype=summary.dtype)
            queries = queries + self.grid_proj(grid).unsqueeze(0)

        if self.decoder_kind in {"tree", "snowflake"}:
            coarse_tokens = self.coarse_token(summary).view(b, self.coarse_points, self.width)
            repeats = math.ceil(self.num_output_points / self.coarse_points)
            expanded = coarse_tokens.repeat_interleave(repeats, dim=1)[
                :, : self.num_output_points, :
            ]
            queries = queries + expanded

        if self.decoder_kind == "anchor":
            anchors = self.anchor_proj(self.anchor_bank.to(dtype=summary.dtype))
            repeats = math.ceil(self.num_output_points / anchors.shape[0])
            anchor_tokens = anchors.unsqueeze(0).expand(b, -1, -1)
            anchor_tokens = anchor_tokens.repeat(1, repeats, 1)[:, : self.num_output_points, :]
            queries = queries + anchor_tokens

        if self.decoder_kind == "text":
            prompt_vec = _prompt_stats(
                prompt,
                batch_size=b,
                device=summary.device,
                dtype=summary.dtype,
            )
            queries = queries + self.prompt_proj(prompt_vec).unsqueeze(1)

        return queries, self.coarse_xyz(summary).view(b, self.coarse_points, 3)

    def _refine_queries(self, queries: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        if self.decoder_kind in {"transformer", "text", "snowflake"}:
            return self.transformer(queries)
        if self.decoder_kind == "state_space":
            return self.ssm(queries.transpose(1, 2)).transpose(1, 2)
        if self.decoder_kind == "diffusion":
            refined = queries
            for _ in range(self.depth):
                refined = refined + 0.5 * self.query_mlp(refined + summary.unsqueeze(1))
            return refined
        return queries + self.query_mlp(queries)

    def complete(
        self,
        points: torch.Tensor,
        *,
        prompt: str | list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        x = check_points(points)
        xyz = x[..., :3]
        summary = self._encode(x)
        queries, coarse = self._query_tokens(summary, prompt=prompt)
        refined = self._refine_queries(queries, summary)

        observed = _repeat_or_sample(xyz, self.num_output_points)
        repeats = math.ceil(self.num_output_points / self.coarse_points)
        coarse = coarse.repeat_interleave(repeats, dim=1)[:, : self.num_output_points, :]
        coarse = coarse + xyz.mean(dim=1, keepdim=True)

        delta = torch.tanh(self.point_head(refined))
        return observed + 0.18 * coarse + 0.28 * delta

    def forward(
        self,
        points: torch.Tensor,
        *,
        prompt: str | list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        return self.complete(points, prompt=prompt)


def build_toy_completer(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    num_output_points: int | None = None,
    encoder_kind: str,
    decoder_kind: str,
    dropout: float = 0.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in variants:
        raise ValueError(f"Unknown {family} variant: {variant!r}. Supported: {sorted(variants)}")
    spec = variants[name]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    out_points = int(spec["points"]) if num_output_points is None else int(num_output_points)
    return ToyPointCloudCompleter(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        num_output_points=out_points,
        encoder_kind=str(encoder_kind),
        decoder_kind=str(decoder_kind),
        dropout=float(dropout),
    )


def smoke_test_completer(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    points = torch.randn(2, 96, 3)
    completed = model(points)
    print(variant, tuple(completed.shape))
    assert completed.ndim == 3 and completed.shape[0] == 2 and completed.shape[-1] == 3
    print("ok")


__all__ = ["ToyPointCloudCompleter", "build_toy_completer", "check_points", "smoke_test_completer"]
