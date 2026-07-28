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


def _hog_cell_histograms(
    x: torch.Tensor,
    *,
    cell_size: int,
    num_bins: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute a simple per-cell HOG feature map.

    Returns: (B, num_bins, Gh, Gw) where Gh=H/cell_size and Gw=W/cell_size.

    Notes:
    - This is a compact-first HOG variant (no block normalization).
    - We use hard binning (non-differentiable w.r.t. angles), which is fine for repo-local smokes.
    """

    x = check_nchw(x)
    x = _to_grayscale(x)

    b, _, h, w = x.shape
    cell = int(cell_size)
    if cell <= 0:
        raise ValueError("cell_size must be > 0")
    if h < cell or w < cell:
        raise ValueError("Input smaller than one HOG cell.")
    if h % cell != 0 or w % cell != 0:
        raise ValueError("Input H and W must be divisible by cell_size for this compact HOG.")

    bins = int(num_bins)
    if bins <= 1:
        raise ValueError("num_bins must be > 1")

    # Sobel-ish gradients.
    kx = torch.tensor([[-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 1, 3)
    ky = torch.tensor([[-1.0], [0.0], [1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 1)
    gx = F.conv2d(x, kx, padding=(0, 1))
    gy = F.conv2d(x, ky, padding=(1, 0))

    mag = torch.sqrt(gx * gx + gy * gy + float(eps))
    ang = torch.atan2(gy, gx).remainder(torch.pi)  # [0, pi)

    # Hard binning: [0, pi) split into `bins` buckets.
    bin_idx = torch.floor(ang / math.pi * float(bins)).to(torch.long).clamp(min=0, max=bins - 1)
    one_hot = F.one_hot(bin_idx.squeeze(1), num_classes=bins).permute(0, 3, 1, 2).to(mag.dtype)
    weighted = one_hot * mag

    # Aggregate per cell via pooling.
    hist = F.avg_pool2d(weighted, kernel_size=cell, stride=cell)

    # Per-cell L2 normalization (compact-first).
    denom = torch.sqrt(hist.square().sum(dim=1, keepdim=True) + float(eps))
    return hist / denom


class HOGSVMDetector(nn.Module):
    """HOG + linear SVM sliding-window detector (compact-first).

    This is a classic pedestrian-detection style pipeline implemented in torch:
    - compute a per-cell HOG feature map
    - apply a learned linear template via conv2d to obtain a score map

    Forward returns:
    - score_map: (B, C, Oh, Ow) where C=num_classes
    - boxes: (B, Oh*Ow, 4) xyxy boxes in pixel coordinates for each score location
    """

    def __init__(
        self,
        *,
        num_classes: int,
        cell_size: int = 8,
        num_bins: int = 9,
        window_cells: tuple[int, int] = (8, 4),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        self.num_classes = nc
        self.cell_size = int(cell_size)
        self.num_bins = int(num_bins)
        self.window_cells = (int(window_cells[0]), int(window_cells[1]))
        self.eps = float(eps)

        wh, ww = self.window_cells
        if wh <= 0 or ww <= 0:
            raise ValueError("window_cells must be positive")

        self.svm_weight = nn.Parameter(torch.randn(nc, self.num_bins, wh, ww) * 0.01)
        self.svm_bias = nn.Parameter(torch.zeros(nc))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        hog = _hog_cell_histograms(
            x,
            cell_size=int(self.cell_size),
            num_bins=int(self.num_bins),
            eps=float(self.eps),
        )

        wh, ww = self.window_cells
        gh, gw = hog.shape[-2], hog.shape[-1]
        out_h = int(gh) - int(wh) + 1
        out_w = int(gw) - int(ww) + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                "Input too small for HOG window. "
                f"Got cells=({gh},{gw}), window_cells=({wh},{ww})."
            )

        score_map = F.conv2d(hog, self.svm_weight, self.svm_bias)

        ys = torch.arange(out_h, device=x.device, dtype=torch.float32)
        xs = torch.arange(out_w, device=x.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        x1 = gx * float(self.cell_size)
        y1 = gy * float(self.cell_size)
        x2 = x1 + float(ww * self.cell_size)
        y2 = y1 + float(wh * self.cell_size)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
        boxes = boxes.unsqueeze(0).expand(int(x.shape[0]), -1, -1).contiguous()

        return {"score_map": score_map, "boxes": boxes}


_VARIANTS: dict[str, dict] = {
    "pedestrian_hog_svm": {"cell": 8, "bins": 9, "win": (8, 4)},
}


def build_pedestrian_hog_svm_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "pedestrian_hog_svm",
    width_mult: float = 1.0,
) -> nn.Module:
    _ = int(in_channels)
    _ = float(width_mult)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown HOG+SVM variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return HOGSVMDetector(
        num_classes=int(num_classes),
        cell_size=int(spec["cell"]),
        num_bins=int(spec["bins"]),
        window_cells=tuple(spec["win"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_pedestrian_hog_svm_detector(
        in_channels=3, num_classes=1, variant="pedestrian_hog_svm", width_mult=0.5
    )
    out = m(x)
    print("pedestrian_hog_svm", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["score_map"].mean() + out["boxes"].mean()
    loss.backward()
    print("ok")
