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

    hist = F.avg_pool2d(weighted, kernel_size=cell, stride=cell)
    denom = torch.sqrt(hist.square().sum(dim=1, keepdim=True) + float(eps))
    return hist / denom


def _quadratic_def_cost(
    *,
    max_disp: int,
    w_dy: torch.Tensor,
    w_dx: torch.Tensor,
) -> torch.Tensor:
    """Quadratic deformation cost for a (2d+1)x(2d+1) window.

    Returns: (C, K) where K=(2d+1)^2 and C=len(w_dy)=len(w_dx).
    """

    d = int(max_disp)
    ys = torch.arange(-d, d + 1, device=w_dy.device, dtype=torch.float32)
    xs = torch.arange(-d, d + 1, device=w_dy.device, dtype=torch.float32)
    dy, dx = torch.meshgrid(ys, xs, indexing="ij")
    dy2 = (dy * dy).reshape(-1)  # (K,)
    dx2 = (dx * dx).reshape(-1)  # (K,)

    w_dy = w_dy.to(torch.float32).view(-1, 1)
    w_dx = w_dx.to(torch.float32).view(-1, 1)
    return w_dy * dy2.view(1, -1) + w_dx * dx2.view(1, -1)


class PedestrianDPMDetector(nn.Module):
    """Deformable Part Model (DPM) style detector (compact-first).

    This is a simplified, torch-native DPM:
    - compute HOG per cell
    - score a root filter with conv2d
    - score several part filters; for each root location, pick the best part displacement
      within a small window and subtract a quadratic deformation cost

    Forward returns:
    - score_map: (B, C, Oh, Ow)
    - boxes: (B, Oh*Ow, 4) xyxy root boxes in pixel coordinates for each score location
    """

    def __init__(
        self,
        *,
        num_classes: int,
        cell_size: int = 8,
        num_bins: int = 9,
        root_cells: tuple[int, int] = (8, 4),
        part_cells: tuple[int, int] = (4, 2),
        part_anchors: tuple[tuple[int, int], ...] = ((0, 0), (0, 2), (4, 0), (4, 2)),
        max_disp: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")

        self.num_classes = nc
        self.cell_size = int(cell_size)
        self.num_bins = int(num_bins)
        self.root_cells = (int(root_cells[0]), int(root_cells[1]))
        self.part_cells = (int(part_cells[0]), int(part_cells[1]))
        self.part_anchors = tuple((int(y), int(x)) for y, x in part_anchors)
        self.max_disp = int(max_disp)
        self.eps = float(eps)

        rh, rw = self.root_cells
        ph, pw = self.part_cells
        if rh <= 0 or rw <= 0 or ph <= 0 or pw <= 0:
            raise ValueError("root_cells and part_cells must be positive")
        if ph > rh or pw > rw:
            raise ValueError("part_cells must be <= root_cells")
        if self.max_disp < 0:
            raise ValueError("max_disp must be >= 0")

        # Root filter.
        self.root_weight = nn.Parameter(torch.randn(nc, self.num_bins, rh, rw) * 0.01)
        self.root_bias = nn.Parameter(torch.zeros(nc))

        # Part filters.
        num_parts = len(self.part_anchors)
        self.part_weight = nn.Parameter(torch.randn(num_parts, nc, self.num_bins, ph, pw) * 0.01)
        self.part_bias = nn.Parameter(torch.zeros(num_parts, nc))

        # Deformation weights (positive) per part/class: cost = w_dy * dy^2 + w_dx * dx^2
        self.def_w_raw = nn.Parameter(torch.zeros(num_parts, nc, 2))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        hog = _hog_cell_histograms(
            x, cell_size=int(self.cell_size), num_bins=int(self.num_bins), eps=float(self.eps)
        )

        # Root score map.
        root_score = F.conv2d(hog, self.root_weight, self.root_bias)  # (B,C,Oh,Ow)
        b, c, out_h, out_w = root_score.shape

        # Precompute root boxes (pixel xyxy) for each root location.
        rh, rw = self.root_cells
        ys = torch.arange(out_h, device=x.device, dtype=torch.float32)
        xs = torch.arange(out_w, device=x.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        x1 = gx * float(self.cell_size)
        y1 = gy * float(self.cell_size)
        x2 = x1 + float(rw * self.cell_size)
        y2 = y1 + float(rh * self.cell_size)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
        boxes = boxes.unsqueeze(0).expand(b, -1, -1).contiguous()

        # Part scoring: for each part, pick best displacement around the anchor.
        ph, pw = self.part_cells
        max_disp = int(self.max_disp)
        k = 2 * max_disp + 1
        k2 = k * k

        total_part_score = torch.zeros(
            (b, c, out_h, out_w), device=x.device, dtype=root_score.dtype
        )

        # Deformation cost template per part/class (C, K).
        def_w = F.softplus(self.def_w_raw)

        for part_idx, (ay, ax) in enumerate(self.part_anchors):
            part_resp = F.conv2d(
                hog,
                self.part_weight[part_idx],
                self.part_bias[part_idx],
            )  # (B,C,Ph_out,Pw_out)
            ph_out, pw_out = int(part_resp.shape[-2]), int(part_resp.shape[-1])

            # Extract a (k x k) window around every possible anchor location using unfold.
            part_padded = F.pad(part_resp, (max_disp, max_disp, max_disp, max_disp), value=-1e9)
            patches = F.unfold(part_padded, kernel_size=k, stride=1)  # (B, C*k2, Ph_out*Pw_out)
            patches = patches.view(b, c, k2, ph_out * pw_out)

            # For each root location (y,x), anchor location in part_resp is (y+ay, x+ax).
            y_idx = torch.arange(out_h, device=x.device, dtype=torch.long).view(out_h, 1)
            x_idx = torch.arange(out_w, device=x.device, dtype=torch.long).view(1, out_w)
            anchor_y = y_idx + int(ay)
            anchor_x = x_idx + int(ax)

            if anchor_y.max().item() >= ph_out or anchor_x.max().item() >= pw_out:
                raise RuntimeError(
                    "Part anchors out of range for current feature map sizes: "
                    f"anchor=({ay},{ax}), part_out=({ph_out},{pw_out}), root_out=({out_h},{out_w})."
                )

            anchor_lin = (anchor_y * pw_out + anchor_x).view(-1)  # (L_root,)
            gathered = patches.index_select(dim=3, index=anchor_lin)  # (B,C,k2,L_root)

            w_dy = def_w[part_idx, :, 0]
            w_dx = def_w[part_idx, :, 1]
            cost = _quadratic_def_cost(max_disp=max_disp, w_dy=w_dy, w_dx=w_dx).to(
                dtype=gathered.dtype
            )  # (C, k2)

            # best over displacements.
            score = gathered - cost.view(1, c, k2, 1)
            best = score.max(dim=2).values  # (B,C,L_root)
            total_part_score = total_part_score + best.view(b, c, out_h, out_w)

        score_map = root_score + total_part_score
        return {"score_map": score_map, "boxes": boxes}


_VARIANTS: dict[str, dict] = {
    "pedestrian_dpm": {
        "cell": 8,
        "bins": 9,
        "root": (8, 4),
        "part": (4, 2),
        "anchors": ((0, 0), (0, 2), (4, 0), (4, 2)),
        "max_disp": 1,
    }
}


def build_pedestrian_dpm_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "pedestrian_dpm",
    width_mult: float = 1.0,
) -> nn.Module:
    _ = int(in_channels)
    _ = float(width_mult)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DPM variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return PedestrianDPMDetector(
        num_classes=int(num_classes),
        cell_size=int(spec["cell"]),
        num_bins=int(spec["bins"]),
        root_cells=tuple(spec["root"]),
        part_cells=tuple(spec["part"]),
        part_anchors=tuple(tuple(v) for v in spec["anchors"]),
        max_disp=int(spec["max_disp"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_pedestrian_dpm_detector(in_channels=3, num_classes=1, variant="pedestrian_dpm")
    out = m(x)
    print("pedestrian_dpm", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["score_map"].mean() + out["boxes"].mean()
    loss.backward()
    print("ok")
