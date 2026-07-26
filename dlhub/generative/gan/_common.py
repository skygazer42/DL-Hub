from __future__ import annotations

import torch
from torch import nn


def _make_mlp(
    *,
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    dropout: float,
    final_tanh: bool,
) -> nn.Module:
    layers: list[nn.Module] = []
    d = max(1, int(depth))
    cur = int(in_dim)
    for _ in range(d):
        layers.append(nn.Linear(cur, int(width)))
        layers.append(nn.GELU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        cur = int(width)
    layers.append(nn.Linear(cur, int(out_dim)))
    if final_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ToyGAN(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        image_size: int,
        latent_dim: int,
        width: int,
        depth: int,
        dropout: float = 0.0,
        num_classes: int = 0,
        use_condition: bool = False,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.num_classes = int(max(0, num_classes))
        self.use_condition = bool(use_condition and self.num_classes > 0)
        self.cond_dim = int(max(8, min(64, int(width) // 2))) if self.use_condition else 0

        self.label_embed: nn.Module | None
        if self.use_condition:
            self.label_embed = nn.Embedding(self.num_classes, self.cond_dim)
        else:
            self.label_embed = None

        flat_dim = self.in_channels * self.image_size * self.image_size
        g_in = self.latent_dim + self.cond_dim
        d_in = flat_dim + self.cond_dim

        self.generator = _make_mlp(
            in_dim=g_in,
            out_dim=flat_dim,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
            final_tanh=True,
        )
        self.discriminator = _make_mlp(
            in_dim=d_in,
            out_dim=1,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
            final_tanh=False,
        )

    def _prepare_labels(
        self,
        *,
        batch_size: int,
        device: torch.device,
        labels: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.use_condition:
            return None
        if labels is None:
            return torch.randint(0, self.num_classes, (int(batch_size),), device=device)
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape {tuple(labels.shape)}")
        if labels.shape[0] != int(batch_size):
            raise ValueError(f"labels batch mismatch: expected {batch_size}, got {labels.shape[0]}")
        return labels.to(device=device, dtype=torch.long)

    def _fuse_with_labels(
        self,
        x: torch.Tensor,
        *,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_condition:
            return x
        assert self.label_embed is not None
        assert labels is not None
        emb = self.label_embed(labels)
        return torch.cat([x, emb], dim=1)

    def sample(
        self,
        *,
        batch_size: int = 4,
        device: torch.device | None = None,
        z: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = int(batch_size) if z is None else int(z.shape[0])
        dev = device if device is not None else torch.device("cpu")
        if z is None:
            z = torch.randn(b, self.latent_dim, device=dev)
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(f"z must have shape (B, {self.latent_dim}), got {tuple(z.shape)}")
        z = z.to(device=dev, dtype=torch.float32)
        cond = self._prepare_labels(batch_size=b, device=z.device, labels=labels)
        g_in = self._fuse_with_labels(z, labels=cond)
        flat = self.generator(g_in)
        return flat.view(b, self.in_channels, self.image_size, self.image_size)

    def discriminate(
        self,
        images: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape (B, C, H, W), " f"got {tuple(images.shape)}")
        b, c, h, w = images.shape
        if int(c) != self.in_channels or int(h) != self.image_size or int(w) != self.image_size:
            raise ValueError(
                "images shape mismatch: expected "
                f"(B,{self.in_channels},{self.image_size},{self.image_size}), "
                f"got {tuple(images.shape)}"
            )
        flat = images.view(int(b), -1).to(torch.float32)
        cond = self._prepare_labels(batch_size=int(b), device=flat.device, labels=labels)
        d_in = self._fuse_with_labels(flat, labels=cond)
        return self.discriminator(d_in).view(-1)

    def forward(
        self,
        *,
        batch_size: int = 2,
        device: torch.device | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        dev = device if device is not None else torch.device("cpu")
        if self.use_condition:
            cond = self._prepare_labels(batch_size=int(batch_size), device=dev, labels=labels)
            fake = self.sample(batch_size=int(batch_size), device=dev, labels=cond)
            fake_logits = self.discriminate(fake, labels=cond)
            assert cond is not None
            return {
                "fake_images": fake,
                "fake_logits": fake_logits,
                "labels": cond.to(torch.float32),
            }
        fake = self.sample(batch_size=int(batch_size), device=dev, labels=None)
        fake_logits = self.discriminate(fake, labels=None)
        return {"fake_images": fake, "fake_logits": fake_logits}


def smoke_test_gan(builder, variant: str) -> None:
    model = builder(
        in_channels=3,
        image_size=32,
        latent_dim=64,
        num_classes=10,
        variant=variant,
        width_mult=0.5,
        dropout=0.0,
    )
    out = model.forward(batch_size=2)
    shapes = {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}
    print(variant, shapes)
    assert "fake_images" in out and "fake_logits" in out
    print("ok")
