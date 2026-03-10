import math

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.detection._common import check_nchw


def _to_grayscale(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    if x.shape[1] == 3:
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x * w).sum(dim=1, keepdim=True)
    return x.mean(dim=1, keepdim=True)


def _acf_feature_map(
    x: torch.Tensor,
    *,
    num_orients: int,
    eps: float,
) -> torch.Tensor:
    """Compute a simple ACF-style channel stack at image resolution.

    Channels:
    - RGB (or best-effort)
    - gradient magnitude
    - oriented gradient channels (hard assignment)

    Returns: (B, C_feat, H, W)
    """

    x = check_nchw(x)
    if x.shape[1] >= 3:
        color = x[:, :3]
    else:
        color = x.repeat(1, 3, 1, 1)

    gray = _to_grayscale(x)

    # Sobel-ish gradients.
    kx = torch.tensor([[-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 1, 3)
    ky = torch.tensor([[-1.0], [0.0], [1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 1)
    gx = F.conv2d(gray, kx, padding=(0, 1))
    gy = F.conv2d(gray, ky, padding=(1, 0))

    mag = torch.sqrt(gx * gx + gy * gy + float(eps))
    ang = torch.atan2(gy, gx).remainder(torch.pi)  # [0, pi)

    bins = int(num_orients)
    if bins <= 1:
        raise ValueError("num_orients must be > 1")

    bin_idx = torch.floor(ang / math.pi * float(bins)).to(torch.long).clamp(min=0, max=bins - 1)
    one_hot = F.one_hot(bin_idx.squeeze(1), num_classes=bins).permute(0, 3, 1, 2).to(mag.dtype)
    oriented = one_hot * mag

    return torch.cat([color, mag, oriented], dim=1)


class PedestrianACFDetector(nn.Module):
    """Aggregated Channel Features (ACF) style detector (toy-first).

    This classic pipeline is implemented in torch:
    - compute a stack of simple channels (RGB + gradient mag + oriented gradients)
    - aggregate by average pooling into cell grid
    - apply a learned linear classifier template via conv2d (sliding window)

    Forward returns:
    - score_map: (B, C, Oh, Ow)
    - boxes: (B, Oh*Ow, 4) xyxy boxes in pixel coordinates for each score location
    """

    def __init__(
        self,
        *,
        num_classes: int,
        cell_size: int = 4,
        num_orients: int = 6,
        window_cells: tuple[int, int] = (16, 8),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")

        self.num_classes = nc
        self.cell_size = int(cell_size)
        self.num_orients = int(num_orients)
        self.window_cells = (int(window_cells[0]), int(window_cells[1]))
        self.eps = float(eps)

        wh, ww = self.window_cells
        if self.cell_size <= 0:
            raise ValueError("cell_size must be > 0")
        if wh <= 0 or ww <= 0:
            raise ValueError("window_cells must be positive")

        # C_feat = 3 (RGB) + 1 (mag) + num_orients
        c_feat = 4 + int(self.num_orients)
        self.clf_weight = nn.Parameter(torch.randn(nc, c_feat, wh, ww) * 0.01)
        self.clf_bias = nn.Parameter(torch.zeros(nc))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape

        cell = int(self.cell_size)
        if h % cell != 0 or w % cell != 0:
            raise ValueError("Input H and W must be divisible by cell_size for this toy ACF.")

        feat = _acf_feature_map(x, num_orients=int(self.num_orients), eps=float(self.eps))
        agg = F.avg_pool2d(feat, kernel_size=cell, stride=cell)

        score_map = F.conv2d(agg, self.clf_weight, self.clf_bias)
        out_h, out_w = int(score_map.shape[-2]), int(score_map.shape[-1])

        wh, ww = self.window_cells
        ys = torch.arange(out_h, device=x.device, dtype=torch.float32)
        xs = torch.arange(out_w, device=x.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        x1 = gx * float(cell)
        y1 = gy * float(cell)
        x2 = x1 + float(ww * cell)
        y2 = y1 + float(wh * cell)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
        boxes = boxes.unsqueeze(0).expand(int(b), -1, -1).contiguous()

        return {"score_map": score_map, "boxes": boxes}


_VARIANTS: dict[str, dict] = {
    "pedestrian_acf": {"cell": 4, "orients": 6, "win": (16, 8)},
}


def build_pedestrian_acf_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "pedestrian_acf",
    width_mult: float = 1.0,
) -> nn.Module:
    _ = int(in_channels)
    _ = float(width_mult)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ACF variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return PedestrianACFDetector(
        num_classes=int(num_classes),
        cell_size=int(spec["cell"]),
        num_orients=int(spec["orients"]),
        window_cells=tuple(spec["win"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_pedestrian_acf_detector(in_channels=3, num_classes=1, variant="pedestrian_acf")
    out = m(x)
    print("pedestrian_acf", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["score_map"].mean() + out["boxes"].mean()
    loss.backward()
    print("ok")

