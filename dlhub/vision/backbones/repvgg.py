import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    w = conv.weight
    bias = (
        torch.zeros(w.size(0), device=w.device, dtype=w.dtype) if conv.bias is None else conv.bias
    )

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    fused_w = w * scale
    fused_b = beta + (bias - mean) * (gamma / std)
    return fused_w, fused_b


def _identity_kernel(
    channels: int, kernel_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    k = torch.zeros((channels, channels, kernel_size, kernel_size), device=device, dtype=dtype)
    center = kernel_size // 2
    for i in range(channels):
        k[i, i, center, center] = 1.0
    return k


class RepVGGBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, stride: int, *, deploy: bool, dropout: float
    ) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.stride = int(stride)
        self.deploy = bool(deploy)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=float(dropout))

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(
                self.in_ch, self.out_ch, kernel_size=3, stride=self.stride, padding=1, bias=True
            )
            self.rbr_dense = None
            self.rbr_1x1 = None
            self.rbr_identity = None
        else:
            self.rbr_reparam = None
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    self.in_ch,
                    self.out_ch,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_ch),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    self.in_ch,
                    self.out_ch,
                    kernel_size=1,
                    stride=self.stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_ch),
            )
            self.rbr_identity = (
                nn.BatchNorm2d(self.in_ch)
                if (self.out_ch == self.in_ch and self.stride == 1)
                else None
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if self.deploy:
            out = self.rbr_reparam(x)
            out = self.relu(out)
            return self.drop(out)

        assert self.rbr_dense is not None and self.rbr_1x1 is not None
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        out = self.relu(out)
        return self.drop(out)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            conv = self.rbr_reparam
            assert conv is not None
            return conv.weight.detach().clone(), conv.bias.detach().clone()

        assert self.rbr_dense is not None and self.rbr_1x1 is not None
        k3, b3 = _fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        k1, b1 = _fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        k1 = torch.nn.functional.pad(k1, [1, 1, 1, 1])

        if self.rbr_identity is not None:
            bn = self.rbr_identity
            kid = _identity_kernel(self.in_ch, 3, device=k3.device, dtype=k3.dtype)
            gamma = bn.weight
            beta = bn.bias
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps
            std = torch.sqrt(var + eps)
            scale = (gamma / std).reshape(-1, 1, 1, 1)
            kid = kid * scale
            bid = beta + (torch.zeros_like(mean) - mean) * (gamma / std)
        else:
            kid = torch.zeros_like(k3)
            bid = torch.zeros_like(b3)

        kernel = k3 + k1 + kid
        bias = b3 + b1 + bid
        return kernel, bias

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_ch, self.out_ch, kernel_size=3, stride=self.stride, padding=1, bias=True
        )
        self.rbr_reparam.weight.data.copy_(kernel)
        self.rbr_reparam.bias.data.copy_(bias)

        self.rbr_dense = None
        self.rbr_1x1 = None
        self.rbr_identity = None
        self.deploy = True


class RepVGGClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float,
        dropout: float,
        stage_blocks: tuple[int, int, int, int] = (1, 2, 4, 1),
        deploy: bool = False,
    ) -> None:
        super().__init__()
        base = scale_channels(32, float(width_mult), min_ch=8, divisor=8)

        def make_stage(in_ch: int, out_ch: int, blocks: int, first_stride: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            for i in range(int(blocks)):
                stride = int(first_stride) if i == 0 else 1
                layers.append(
                    RepVGGBlock(
                        in_ch=int(in_ch) if i == 0 else int(out_ch),
                        out_ch=int(out_ch),
                        stride=stride,
                        deploy=bool(deploy),
                        dropout=float(dropout),
                    )
                )
            return nn.Sequential(*layers)

        self.stage0 = RepVGGBlock(
            in_ch=int(in_channels),
            out_ch=base,
            stride=1,
            deploy=bool(deploy),
            dropout=float(dropout),
        )
        self.stage1 = make_stage(base, base, blocks=int(stage_blocks[0]), first_stride=1)
        self.stage2 = make_stage(base, base * 2, blocks=int(stage_blocks[1]), first_stride=2)
        self.stage3 = make_stage(base * 2, base * 4, blocks=int(stage_blocks[2]), first_stride=2)
        self.stage4 = make_stage(base * 4, base * 8, blocks=int(stage_blocks[3]), first_stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base * 8, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def switch_to_deploy(self) -> None:
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()


def build_repvgg_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "a0",
    width_mult: float = 1.0,
    dropout: float = 0.0,
    deploy: bool = False,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name in {"repvgg_a0", "a0"}:
        stage_blocks = (1, 2, 4, 1)
    elif name in {"repvgg_a1", "a1"}:
        stage_blocks = (1, 2, 6, 2)
    elif name in {"repvgg_a2", "a2"}:
        stage_blocks = (1, 3, 8, 1)
    elif name in {"repvgg_b0", "b0"}:
        stage_blocks = (1, 4, 6, 1)
    elif name in {"repvgg_b1", "b1"}:
        stage_blocks = (2, 4, 6, 2)
    elif name in {"repvgg_b2", "b2"}:
        stage_blocks = (2, 4, 8, 2)
    elif name in {"repvgg_b3", "b3"}:
        stage_blocks = (2, 6, 10, 2)
    else:
        raise ValueError(f"Unknown RepVGG variant: {variant!r}. Supported: a0|a1|a2|b0|b1|b2|b3")

    return RepVGGClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
        stage_blocks=tuple(map(int, stage_blocks)),
        deploy=bool(deploy),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["a0", "a1", "b0"]:
        m = build_repvgg_classifier(
            in_channels=3, num_classes=10, variant=v, width_mult=0.75, deploy=False
        )
        y = m(x)
        print(f"repvgg_{v}", tuple(y.shape))
