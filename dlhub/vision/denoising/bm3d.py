
import math

import torch
from torch import nn
import torch.nn.functional as F


def _dct_matrix(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal DCT-II matrix (so inverse is transpose)."""

    n = int(n)
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # (n, 1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)  # (1, n)
    mat = torch.cos(math.pi * (2.0 * i + 1.0) * k / (2.0 * float(n)))  # (n, n)
    mat[0] = mat[0] / math.sqrt(2.0)
    mat = mat * math.sqrt(2.0 / float(n))
    return mat


def _dct2(patches: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # patches: (G, P, P) ; c: (P, P)
    tmp = torch.einsum("ij,gjk->gik", c, patches)
    return torch.einsum("gij,jk->gik", tmp, c.t())


def _idct2(coeffs: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # Inverse for orthonormal DCT: C^T @ X @ C
    tmp = torch.einsum("ij,gjk->gik", c.t(), coeffs)
    return torch.einsum("gij,jk->gik", tmp, c)


def _dct_group(coeffs: torch.Tensor, cg: torch.Tensor) -> torch.Tensor:
    # coeffs: (G, P, P), cg: (G, G)
    return torch.einsum("ij,jkl->ikl", cg, coeffs)


def _idct_group(coeffs: torch.Tensor, cg: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,jkl->ikl", cg.t(), coeffs)


def _pad_for_patches(x: torch.Tensor, *, patch: int, step: int) -> tuple[torch.Tensor, int, int]:
    """Pad bottom/right so unfold() covers the image on a stride grid."""

    if x.ndim != 2:
        raise ValueError(f"Expected 2D (H, W) input, got {tuple(x.shape)}")

    h, w = x.shape
    patch = int(patch)
    step = int(step)
    if patch <= 0 or step <= 0:
        raise ValueError("patch and step must be positive")

    pad_h = max(0, patch - h)
    pad_w = max(0, patch - w)

    hh = h + pad_h
    ww = w + pad_w
    # Make (size - patch) divisible by step.
    pad_h += (step - ((hh - patch) % step)) % step
    pad_w += (step - ((ww - patch) % step)) % step

    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    # replicate is robust even when padding is large.
    y = F.pad(x[None, None], (0, pad_w, 0, pad_h), mode="replicate")[0, 0]
    return y, int(pad_h), int(pad_w)


def _unfold_patches(x: torch.Tensor, *, patch: int, step: int) -> tuple[torch.Tensor, int, int]:
    """Return (patches, nH, nW), patches shape (L, P, P)."""

    x_pad, _, _ = _pad_for_patches(x, patch=int(patch), step=int(step))
    h, w = x_pad.shape
    patch = int(patch)
    step = int(step)
    n_h = (h - patch) // step + 1
    n_w = (w - patch) // step + 1

    cols = F.unfold(x_pad[None, None], kernel_size=patch, stride=step)  # (1, P*P, L)
    patches = cols[0].t().reshape(n_h * n_w, patch, patch)  # (L, P, P)
    return patches, int(n_h), int(n_w)


def _fold_accumulate(
    *,
    out_sum: torch.Tensor,
    out_w: torch.Tensor,
    patch: torch.Tensor,
    top: int,
    left: int,
    weight: float,
) -> None:
    p = patch.shape[-1]
    w = float(weight)
    out_sum[top : top + p, left : left + p] += patch * w
    out_w[top : top + p, left : left + p] += w


def _bm3d_stage(
    noisy: torch.Tensor,
    *,
    ref: torch.Tensor | None,
    sigma: float,
    patch_size: int,
    step: int,
    search_radius: int,
    max_group_size: int,
    lambda3d: float,
) -> torch.Tensor:
    """One BM3D stage over a single-channel 2D image.

    Args:
        noisy: (H, W)
        ref: None for stage-1; for stage-2 this should be the basic estimate (H, W).
    """

    device = noisy.device
    dtype = noisy.dtype

    p = int(patch_size)
    s = int(step)
    sr = int(search_radius)
    gmax = int(max_group_size)
    sig = float(sigma)
    thr = float(lambda3d) * sig

    c_p = _dct_matrix(p, device=device, dtype=dtype)

    noisy_pad, pad_h, pad_w = _pad_for_patches(noisy, patch=p, step=s)
    if ref is None:
        ref_pad = noisy_pad
    else:
        ref_pad, _, _ = _pad_for_patches(ref, patch=p, step=s)

    h_pad, w_pad = noisy_pad.shape
    n_h = (h_pad - p) // s + 1
    n_w = (w_pad - p) // s + 1

    noisy_patches, _, _ = _unfold_patches(noisy_pad, patch=p, step=s)
    ref_patches, _, _ = _unfold_patches(ref_pad, patch=p, step=s)

    out_sum = torch.zeros((h_pad, w_pad), device=device, dtype=dtype)
    out_w = torch.zeros((h_pad, w_pad), device=device, dtype=dtype)

    # Cache small group DCT matrices on this device/dtype.
    group_cache: dict[int, torch.Tensor] = {}

    def get_cg(size: int) -> torch.Tensor:
        size = int(size)
        cg = group_cache.get(size)
        if cg is None:
            cg = _dct_matrix(size, device=device, dtype=dtype)
            group_cache[size] = cg
        return cg

    # Iterate reference patches on the stride grid.
    for ref_idx in range(n_h * n_w):
        i = ref_idx // n_w
        j = ref_idx % n_w

        i0 = max(0, i - sr)
        i1 = min(n_h - 1, i + sr)
        j0 = max(0, j - sr)
        j1 = min(n_w - 1, j + sr)

        cand: list[int] = []
        for ii in range(i0, i1 + 1):
            base = ii * n_w
            for jj in range(j0, j1 + 1):
                cand.append(base + jj)

        ref_vec = ref_patches[ref_idx].reshape(1, -1)  # (1, P*P)
        cand_idx = torch.tensor(cand, device=device, dtype=torch.long)
        cand_vec = ref_patches[cand_idx].reshape(len(cand), -1)
        dist = (cand_vec - ref_vec).pow(2).mean(dim=1)  # (K,)

        k = min(gmax, dist.numel())
        best = torch.topk(dist, k=k, largest=False).indices
        group_idx = cand_idx[best]  # (G,)
        g = int(group_idx.numel())
        cg = get_cg(g)

        # Collaborative filtering in transform domain.
        group_noisy = noisy_patches[group_idx]  # (G, P, P)
        group_ref = ref_patches[group_idx]  # (G, P, P)

        noisy_c = _dct_group(_dct2(group_noisy, c_p), cg)
        ref_c = _dct_group(_dct2(group_ref, c_p), cg)

        if ref is None:
            # Stage 1: hard threshold.
            mask = noisy_c.abs() > thr
            filtered_c = noisy_c * mask
            nz = int(mask.sum().item())
            weight = 1.0 / float(max(1, nz))
        else:
            # Stage 2: Wiener shrinkage based on basic estimate (ref_c).
            power = ref_c.pow(2)
            wiener = power / (power + sig * sig)
            filtered_c = noisy_c * wiener
            weight = 1.0 / float(wiener.sum().clamp_min(1e-8).item())

        group_f = _idct2(_idct_group(filtered_c, cg), c_p)  # (G, P, P)

        # Aggregate patches back.
        for t, patch_f in zip(group_idx.tolist(), group_f, strict=True):
            top = (t // n_w) * s
            left = (t % n_w) * s
            _fold_accumulate(out_sum=out_sum, out_w=out_w, patch=patch_f, top=top, left=left, weight=weight)

    out = out_sum / out_w.clamp_min(1e-8)
    # Crop to pre-padding spatial size.
    if pad_h or pad_w:
        out = out[: h_pad - pad_h, : w_pad - pad_w]
    return out


class BM3D(nn.Module):
    """A small, torch-only BM3D implementation (educational / toy-first).

    Important:
    - This is a simplified BM3D: it implements the core block-matching + collaborative filtering idea
      with orthonormal DCTs. It is intended for small images (e.g., 32-128px) and CPU baselines.
    - It is not optimized and uses Python loops (acceptable for lessons / toy datasets).
    """

    def __init__(
        self,
        *,
        sigma: float = 0.1,
        patch_size: int = 8,
        step: int = 4,
        search_radius: int = 3,
        max_group_size: int = 16,
        lambda3d: float = 2.7,
        stages: int = 2,
        clamp: bool = True,
    ) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.patch_size = int(patch_size)
        self.step = int(step)
        self.search_radius = int(search_radius)
        self.max_group_size = int(max_group_size)
        self.lambda3d = float(lambda3d)
        self.stages = int(stages)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        with torch.no_grad():
            b, c, h, w = x.shape
            out = torch.empty_like(x)
            for bi in range(b):
                for ci in range(c):
                    noisy = x[bi, ci]
                    basic = _bm3d_stage(
                        noisy,
                        ref=None,
                        sigma=self.sigma,
                        patch_size=self.patch_size,
                        step=self.step,
                        search_radius=self.search_radius,
                        max_group_size=self.max_group_size,
                        lambda3d=self.lambda3d,
                    )
                    if self.stages >= 2:
                        final = _bm3d_stage(
                            noisy,
                            ref=basic,
                            sigma=self.sigma,
                            patch_size=self.patch_size,
                            step=self.step,
                            search_radius=self.search_radius,
                            max_group_size=self.max_group_size,
                            lambda3d=self.lambda3d,
                        )
                    else:
                        final = basic

                    if self.clamp:
                        final = final.clamp(0.0, 1.0)
                    out[bi, ci] = final
            return out


_VARIANTS: dict[str, dict] = {
    "bm3d_fast": {"patch_size": 8, "step": 4, "search_radius": 2, "max_group_size": 12, "lambda3d": 2.7, "stages": 2},
    "bm3d_quality": {"patch_size": 8, "step": 3, "search_radius": 3, "max_group_size": 16, "lambda3d": 2.7, "stages": 2},
    "bm3d_stage1": {"patch_size": 8, "step": 4, "search_radius": 2, "max_group_size": 12, "lambda3d": 2.7, "stages": 1},
}


def build_bm3d_denoiser(
    *,
    in_channels: int,  # kept for consistent builder signatures (unused by BM3D).
    sigma: float = 0.1,
    variant: str = "bm3d_fast",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BM3D variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return BM3D(
        sigma=float(sigma),
        patch_size=int(spec["patch_size"]),
        step=int(spec["step"]),
        search_radius=int(spec["search_radius"]),
        max_group_size=int(spec["max_group_size"]),
        lambda3d=float(spec["lambda3d"]),
        stages=int(spec["stages"]),
        clamp=True,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(1, 1, 64, 64)
    noise = torch.randn_like(x) * 0.1
    noisy = (x + noise).clamp(0.0, 1.0)

    m = build_bm3d_denoiser(in_channels=1, sigma=0.1, variant="bm3d_fast")
    y = m(noisy)
    print("bm3d_fast", tuple(y.shape), float(((y - x).pow(2).mean()).item()))

