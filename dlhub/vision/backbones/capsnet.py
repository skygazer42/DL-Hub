import torch
from torch import nn


def _squash(x: torch.Tensor, *, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    # Squashing non-linearity for capsules
    sq_norm = torch.sum(x * x, dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * x / torch.sqrt(sq_norm + float(eps))


class PrimaryCapsules(nn.Module):
    def __init__(
        self,
        in_ch: int,
        *,
        num_capsules: int,
        capsule_dim: int,
        grid_size: int,
    ) -> None:
        super().__init__()
        n = int(num_capsules)
        d = int(capsule_dim)
        g = int(grid_size)
        if n <= 0 or d <= 0 or g <= 0:
            raise ValueError("num_capsules, capsule_dim, grid_size must be > 0")
        self.num_capsules = n
        self.capsule_dim = d
        self.grid_size = g

        self.conv = nn.Conv2d(int(in_ch), n * d, kernel_size=1, stride=1, padding=0, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((g, g))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        x = self.pool(x)
        x = self.conv(x)  # (B, N*D, g, g)
        b, _, g1, g2 = x.shape
        x = x.view(b, self.num_capsules, self.capsule_dim, g1, g2)
        x = (
            x.permute(0, 3, 4, 1, 2)
            .contiguous()
            .view(b, g1 * g2 * self.num_capsules, self.capsule_dim)
        )
        return _squash(x, dim=-1)


class DigitCapsules(nn.Module):
    def __init__(
        self,
        num_in_caps: int,
        in_dim: int,
        num_out_caps: int,
        out_dim: int,
        *,
        routing_iters: int = 3,
    ) -> None:
        super().__init__()
        ni = int(num_in_caps)
        no = int(num_out_caps)
        di = int(in_dim)
        do = int(out_dim)
        r = int(routing_iters)
        if ni <= 0 or no <= 0 or di <= 0 or do <= 0:
            raise ValueError("capsule counts/dims must be > 0")
        if r <= 0:
            raise ValueError("routing_iters must be > 0")
        self.num_in_caps = ni
        self.num_out_caps = no
        self.in_dim = di
        self.out_dim = do
        self.routing_iters = r

        # W: (1, No, Ni, Do, Di)
        self.W = nn.Parameter(0.01 * torch.randn(1, no, ni, do, di))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Ni, Di)
        b, ni, di = x.shape
        if ni != self.num_in_caps or di != self.in_dim:
            raise ValueError(f"Expected (B,{self.num_in_caps},{self.in_dim}), got {tuple(x.shape)}")

        x = x.unsqueeze(1).unsqueeze(-1)  # (B, 1, Ni, Di, 1)
        W = self.W.expand(b, -1, -1, -1, -1)  # (B, No, Ni, Do, Di)
        u_hat = torch.matmul(W, x).squeeze(-1)  # (B, No, Ni, Do)

        b_ij = torch.zeros(
            b, self.num_out_caps, self.num_in_caps, device=x.device, dtype=u_hat.dtype
        )
        for _ in range(self.routing_iters):
            c_ij = torch.softmax(b_ij, dim=1)  # (B, No, Ni)
            s_j = torch.sum(c_ij.unsqueeze(-1) * u_hat, dim=2)  # (B, No, Do)
            v_j = _squash(s_j, dim=-1)  # (B, No, Do)
            # agreement
            b_ij = b_ij + torch.sum(u_hat * v_j.unsqueeze(2), dim=-1)
        return v_j


class CapsNetClassifier(nn.Module):
    """Capsule Network (Sabour et al.) simplified for arbitrary image sizes."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        grid_size: int = 6,
        primary_caps: int = 8,
        primary_dim: int = 8,
        digit_dim: int = 16,
        routing_iters: int = 3,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(int(in_channels), 64, kernel_size=5, stride=2, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.primary = PrimaryCapsules(
            256,
            num_capsules=int(primary_caps),
            capsule_dim=int(primary_dim),
            grid_size=int(grid_size),
        )
        num_in_caps = int(grid_size) * int(grid_size) * int(primary_caps)
        self.digit = DigitCapsules(
            num_in_caps=num_in_caps,
            in_dim=int(primary_dim),
            num_out_caps=int(num_classes),
            out_dim=int(digit_dim),
            routing_iters=int(routing_iters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        x = self.primary(x)  # (B, Ni, Di)
        v = self.digit(x)  # (B, num_classes, digit_dim)
        # Class logits as capsule lengths
        return torch.sqrt(torch.sum(v * v, dim=-1) + 1e-8)


_VARIANTS: dict[str, dict] = {
    "capsnet_small": {"grid": 5, "primary_caps": 6, "primary_dim": 8, "digit_dim": 16},
    "capsnet_base": {"grid": 6, "primary_caps": 8, "primary_dim": 8, "digit_dim": 16},
    "capsnet_large": {"grid": 8, "primary_caps": 8, "primary_dim": 8, "digit_dim": 24},
}


def build_capsnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "capsnet_base",
    routing_iters: int = 3,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CapsNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return CapsNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        grid_size=int(spec["grid"]),
        primary_caps=int(spec["primary_caps"]),
        primary_dim=int(spec["primary_dim"]),
        digit_dim=int(spec["digit_dim"]),
        routing_iters=int(routing_iters),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["capsnet_small", "capsnet_base"]:
        m = build_capsnet_classifier(in_channels=3, num_classes=10, variant=v, routing_iters=3)
        y = m(x)
        print(v, tuple(y.shape))
