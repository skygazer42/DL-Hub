from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 32
    image_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    in_channels: int = 1
    # Noise models (toy-first). `noise_std` is used by Gaussian and as the base scale for some hybrids.
    noise_type: str = "gaussian"  # gaussian | gaussian_impulse | poisson | impulse | shot_read | speckle | stripe
    noise_std: float = 0.1  # Gaussian std in [0,1] scale
    poisson_peak: float = 30.0  # Peak photons for Poisson noise (higher = less noise)
    impulse_prob: float = 0.03  # Salt & pepper probability
    shot_noise: float = 0.2  # Heteroscedastic term (variance ~ shot_noise * signal)
    read_noise: float = 0.02  # Additive read noise (std in [0,1] scale)
    speckle_std: float = 0.15  # Multiplicative noise scale (used when noise-type=speckle)
    stripe_amplitude: float = 0.12  # Stripe noise amplitude (used when noise-type=stripe)
    stripe_period: int = 8  # Stripe spatial period in pixels (used when noise-type=stripe)
    stripe_direction: str = "vertical"  # vertical | horizontal
    min_square: int = 8
    max_square: int = 24
    train_mode: str = "supervised"  # supervised | noise2noise | blindspot
    blindspot_prob: float = 0.1  # fraction of pixels to mask for blind-spot training


class ToyDenoisingSquares(Dataset):
    """Toy denoising dataset (clean squares + additive Gaussian noise)."""

    def __init__(self, cfg: DataConfig, *, mode: str) -> None:
        self.cfg = cfg
        self.mode = str(mode).lower().strip()
        if self.mode not in {"supervised", "noise2noise", "blindspot"}:
            raise ValueError(f"Unknown mode: {mode!r}. Expected 'supervised' | 'noise2noise' | 'blindspot'.")

        s = int(cfg.image_size)
        if s < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.min_square) < 2 or int(cfg.max_square) < int(cfg.min_square):
            raise ValueError("invalid square size range")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")

        n = int(cfg.num_samples)
        g = torch.Generator().manual_seed(int(cfg.seed))

        sizes = torch.randint(
            low=int(cfg.min_square),
            high=int(cfg.max_square) + 1,
            size=(n,),
            generator=g,
            dtype=torch.long,
        )

        tops = torch.empty((n,), dtype=torch.long)
        lefts = torch.empty((n,), dtype=torch.long)
        for i in range(n):
            size = int(sizes[i].item())
            top = int(torch.randint(low=0, high=max(1, s - size + 1), size=(1,), generator=g).item())
            left = int(torch.randint(low=0, high=max(1, s - size + 1), size=(1,), generator=g).item())
            tops[i] = top
            lefts[i] = left

        self.sizes = sizes
        self.tops = tops
        self.lefts = lefts

    def __len__(self) -> int:
        return int(self.sizes.numel())

    def _clean(self, idx: int) -> torch.Tensor:
        cfg = self.cfg
        s = int(cfg.image_size)
        c = int(cfg.in_channels)
        size = int(self.sizes[idx].item())
        top = int(self.tops[idx].item())
        left = int(self.lefts[idx].item())

        img = torch.zeros((1, s, s), dtype=torch.float32)
        img[:, top : top + size, left : left + size] = 1.0
        if c == 1:
            return img
        return img.repeat(c, 1, 1)

    def _generator(self, idx: int, *, stream: int, salt: int) -> torch.Generator:
        # Per-sample RNG: deterministic across dataloader workers.
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 17 + int(stream) * 101 + int(salt)
        return torch.Generator().manual_seed(seed)

    def _make_noisy(self, clean: torch.Tensor, idx: int, *, stream: int) -> torch.Tensor:
        """Apply the configured noise model to a clean image (C,H,W) -> noisy (C,H,W)."""

        if clean.ndim != 3:
            raise ValueError(f"Expected clean tensor shape (C, H, W), got {tuple(clean.shape)}")

        cfg = self.cfg
        noise_type = str(cfg.noise_type).lower().strip()
        c, h, w = clean.shape

        if noise_type in {"gaussian", "normal"}:
            g = self._generator(idx, stream=stream, salt=0)
            noise = torch.randn((c, h, w), generator=g, dtype=torch.float32) * float(cfg.noise_std)
            return (clean + noise).clamp(0.0, 1.0)

        if noise_type in {"gaussian_impulse", "gaussian+impulse", "gauss_impulse"}:
            # Apply Gaussian, then salt & pepper.
            g = self._generator(idx, stream=stream, salt=0)
            noise = torch.randn((c, h, w), generator=g, dtype=torch.float32) * float(cfg.noise_std)
            out = (clean + noise).clamp(0.0, 1.0)

            p = float(cfg.impulse_prob)
            if not (0.0 <= p < 1.0):
                raise ValueError("impulse_prob must be in [0, 1)")
            if p == 0.0:
                return out
            g2 = self._generator(idx, stream=stream, salt=2)
            u = torch.rand((1, h, w), generator=g2, dtype=torch.float32)
            salt = u < (p * 0.5)
            pepper = (u >= (p * 0.5)) & (u < p)
            if c != 1:
                salt = salt.repeat(c, 1, 1)
                pepper = pepper.repeat(c, 1, 1)
            out = out.clone()
            out[salt] = 1.0
            out[pepper] = 0.0
            return out.clamp(0.0, 1.0)

        if noise_type in {"poisson"}:
            peak = float(cfg.poisson_peak)
            if peak <= 0:
                raise ValueError("poisson_peak must be > 0")
            rate = (clean.clamp(0.0, 1.0) * peak).clamp_min(0.0)
            g = self._generator(idx, stream=stream, salt=1)
            noisy = torch.poisson(rate, generator=g) / peak
            return noisy.clamp(0.0, 1.0)

        if noise_type in {"impulse", "saltpepper", "salt_pepper"}:
            p = float(cfg.impulse_prob)
            if not (0.0 <= p < 1.0):
                raise ValueError("impulse_prob must be in [0, 1)")
            if p == 0.0:
                return clean
            g = self._generator(idx, stream=stream, salt=2)
            u = torch.rand((1, h, w), generator=g, dtype=torch.float32)
            salt = u < (p * 0.5)
            pepper = (u >= (p * 0.5)) & (u < p)
            if c != 1:
                salt = salt.repeat(c, 1, 1)
                pepper = pepper.repeat(c, 1, 1)
            noisy = clean.clone()
            noisy[salt] = 1.0
            noisy[pepper] = 0.0
            return noisy.clamp(0.0, 1.0)

        if noise_type in {"shot_read", "shot+read", "heteroscedastic"}:
            shot = float(cfg.shot_noise)
            read = float(cfg.read_noise)
            if shot < 0.0:
                raise ValueError("shot_noise must be >= 0")
            if read < 0.0:
                raise ValueError("read_noise must be >= 0")
            g = self._generator(idx, stream=stream, salt=3)
            std = torch.sqrt(clean.clamp_min(0.0) * shot + (read * read))
            noise = torch.randn((c, h, w), generator=g, dtype=torch.float32) * std
            return (clean + noise).clamp(0.0, 1.0)

        if noise_type in {"speckle"}:
            sstd = float(cfg.speckle_std)
            if sstd < 0.0:
                raise ValueError("speckle_std must be >= 0")
            g = self._generator(idx, stream=stream, salt=4)
            noise = torch.randn((c, h, w), generator=g, dtype=torch.float32) * sstd
            return (clean + clean * noise).clamp(0.0, 1.0)

        if noise_type in {"stripe", "stripes", "banding"}:
            amp = float(cfg.stripe_amplitude)
            if amp < 0.0:
                raise ValueError("stripe_amplitude must be >= 0")
            period = int(cfg.stripe_period)
            if period <= 1:
                raise ValueError("stripe_period must be > 1")
            direction = str(cfg.stripe_direction).lower().strip()
            if direction not in {"vertical", "horizontal"}:
                raise ValueError("stripe_direction must be 'vertical' or 'horizontal'")

            g = self._generator(idx, stream=stream, salt=5)
            phase = float(torch.rand((), generator=g).item()) * 2.0 * 3.141592653589793

            if direction == "vertical":
                coord = torch.arange(w, dtype=torch.float32)[None, None, :]  # (1,1,W)
                stripe = torch.sin(2.0 * 3.141592653589793 * coord / float(period) + phase)
                stripe = stripe.expand(1, h, w)  # (1,H,W)
            else:
                coord = torch.arange(h, dtype=torch.float32)[None, :, None]  # (1,H,1)
                stripe = torch.sin(2.0 * 3.141592653589793 * coord / float(period) + phase)
                stripe = stripe.expand(1, h, w)  # (1,H,W)

            if c != 1:
                stripe = stripe.repeat(c, 1, 1)
            return (clean + amp * stripe).clamp(0.0, 1.0)

        raise ValueError(
            f"Unknown noise_type: {cfg.noise_type!r}. Supported: gaussian | gaussian_impulse | poisson | impulse | shot_read | speckle | stripe"
        )

    def _blindspot_mask(self, idx: int) -> torch.Tensor:
        cfg = self.cfg
        s = int(cfg.image_size)
        c = int(cfg.in_channels)
        p = float(cfg.blindspot_prob)
        if not (0.0 < p < 1.0):
            raise ValueError("blindspot_prob must be in (0, 1)")

        g = self._generator(idx, stream=0, salt=10)
        m = (torch.rand((1, s, s), generator=g, dtype=torch.float32) < p).to(torch.float32)
        if c == 1:
            return m
        return m.repeat(c, 1, 1)

    def _blindspot_masked_input(self, noisy: torch.Tensor, idx: int) -> torch.Tensor:
        """Noise2Void-style masking: replace masked pixels with random neighbor pixels."""

        if noisy.ndim != 3:
            raise ValueError(f"Expected (C, H, W) noisy tensor, got {tuple(noisy.shape)}")

        mask = self._blindspot_mask(idx)  # (C, H, W)
        if mask.shape != noisy.shape:
            raise ValueError("blindspot mask shape mismatch")

        # Randomly choose replacement direction per pixel (shared across channels).
        cfg = self.cfg
        s = int(cfg.image_size)
        g = self._generator(idx, stream=0, salt=11)
        sel = torch.randint(0, 4, (1, s, s), generator=g, dtype=torch.long)

        up = torch.roll(noisy, shifts=1, dims=1)
        down = torch.roll(noisy, shifts=-1, dims=1)
        left = torch.roll(noisy, shifts=1, dims=2)
        right = torch.roll(noisy, shifts=-1, dims=2)

        rep = torch.where(sel == 0, up, torch.where(sel == 1, down, torch.where(sel == 2, left, right)))
        masked = noisy * (1.0 - mask) + rep * mask
        return masked

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        i = int(idx)
        clean = self._clean(i)

        if self.mode == "supervised":
            noisy = self._make_noisy(clean, i, stream=0)
            return noisy, clean

        # noise2noise: two independent noise realizations of the same clean signal
        noisy1 = self._make_noisy(clean, i, stream=0)
        noisy2 = self._make_noisy(clean, i, stream=1)
        if self.mode == "noise2noise":
            return noisy1, noisy2

        # blindspot: self-supervised via masking (target is the *noisy* image; loss only on masked pixels)
        noisy = noisy1
        masked_noisy = self._blindspot_masked_input(noisy, i)
        mask = self._blindspot_mask(i)
        return masked_noisy, {"target": noisy, "mask": mask}


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    """Return `(train_loader, val_loader)` for the toy denoising task."""

    train_mode = str(cfg.train_mode).lower().strip()
    train_ds_full = ToyDenoisingSquares(cfg, mode=train_mode)
    val_ds_full = ToyDenoisingSquares(cfg, mode="supervised")

    train_idx, val_idx = train_val_split_indices(n=len(train_ds_full), val_fraction=cfg.val_fraction, seed=cfg.seed)
    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(val_ds_full, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "ToyDenoisingSquares", "get_dataloaders"]
