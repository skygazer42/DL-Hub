from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath


@dataclass(frozen=True)
class Genotype:
    normal: tuple[tuple[str, int], ...]
    normal_concat: tuple[int, ...]
    reduce: tuple[tuple[str, int], ...]
    reduce_concat: tuple[int, ...]


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Zero(nn.Module):
    def __init__(self, *, stride: int) -> None:
        super().__init__()
        s = int(stride)
        if s <= 0:
            raise ValueError("stride must be > 0")
        self.stride = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, :: self.stride, :: self.stride].mul(0.0)


class ReLUConvBN(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(int(in_ch), int(out_ch), int(kernel_size), stride=int(stride), padding=int(padding), bias=False),
            nn.BatchNorm2d(int(out_ch)),
        )


class FactorizedReduce(nn.Module):
    """DARTS-style factorized reduction (simplified but shape-safe)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        if c_out <= 0:
            raise ValueError("out_ch must be > 0")
        c1 = c_out // 2
        c2 = c_out - c1
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(c_in, c1, kernel_size=1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(c_in, c2, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        y1 = self.conv1(x)
        y2 = self.conv2(x[:, :, 1:, 1:])
        y = torch.cat([y1, y2], dim=1)
        return self.bn(y)


class PoolBN(nn.Sequential):
    def __init__(self, pool: nn.Module, channels: int) -> None:
        super().__init__(pool, nn.BatchNorm2d(int(channels)))


class DilConv(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ) -> None:
        c_in = int(in_ch)
        c_out = int(out_ch)
        k = int(kernel_size)
        s = int(stride)
        p = int(padding)
        d = int(dilation)
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=k, stride=s, padding=p, dilation=d, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out),
        )


class SepConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, stride: int, padding: int) -> None:
        c_in = int(in_ch)
        c_out = int(out_ch)
        k = int(kernel_size)
        s = int(stride)
        p = int(padding)
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(c_in, c_in, kernel_size=k, stride=s, padding=p, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=False),
            nn.Conv2d(c_out, c_out, kernel_size=k, stride=1, padding=p, groups=c_out, bias=False),
            nn.Conv2d(c_out, c_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out),
        )


def _conv3x3(c: int, stride: int) -> nn.Module:
    return ReLUConvBN(c, c, kernel_size=3, stride=int(stride), padding=1)


def _conv1x1(c: int, stride: int) -> nn.Module:
    return ReLUConvBN(c, c, kernel_size=1, stride=int(stride), padding=0)


OPS: dict[str, Callable[[int, int], nn.Module]] = {
    "none": lambda c, s: Zero(stride=int(s)),
    "skip_connect": lambda c, s: Identity() if int(s) == 1 else FactorizedReduce(int(c), int(c)),
    "max_pool_3x3": lambda c, s: PoolBN(nn.MaxPool2d(3, stride=int(s), padding=1), int(c)),
    "avg_pool_3x3": lambda c, s: PoolBN(nn.AvgPool2d(3, stride=int(s), padding=1, count_include_pad=False), int(c)),
    "sep_conv_3x3": lambda c, s: SepConv(int(c), int(c), kernel_size=3, stride=int(s), padding=1),
    "sep_conv_5x5": lambda c, s: SepConv(int(c), int(c), kernel_size=5, stride=int(s), padding=2),
    "dil_conv_3x3": lambda c, s: DilConv(int(c), int(c), kernel_size=3, stride=int(s), padding=2, dilation=2),
    "dil_conv_5x5": lambda c, s: DilConv(int(c), int(c), kernel_size=5, stride=int(s), padding=4, dilation=2),
    "conv_3x3": lambda c, s: _conv3x3(int(c), int(s)),
    "conv_1x1": lambda c, s: _conv1x1(int(c), int(s)),
}


class NASCell(nn.Module):
    def __init__(
        self,
        genotype: Genotype,
        *,
        c_prev_prev: int,
        c_prev: int,
        c_cur: int,
        reduction: bool,
        reduction_prev: bool,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.reduction = bool(reduction)
        self.reduction_prev = bool(reduction_prev)
        self.drop_path = DropPath(float(drop_path))

        c_pp = int(c_prev_prev)
        c_p = int(c_prev)
        c_c = int(c_cur)

        self.pre0 = (
            FactorizedReduce(c_pp, c_c)
            if self.reduction_prev
            else ReLUConvBN(c_pp, c_c, kernel_size=1, stride=1, padding=0)
        )
        self.pre1 = ReLUConvBN(c_p, c_c, kernel_size=1, stride=1, padding=0)

        if self.reduction:
            ops = tuple(genotype.reduce)
            concat = tuple(genotype.reduce_concat)
        else:
            ops = tuple(genotype.normal)
            concat = tuple(genotype.normal_concat)

        if len(ops) % 2 != 0:
            raise ValueError("Cell op list must contain pairs of (op, idx)")
        self.steps = len(ops) // 2
        self.concat = concat
        self.multiplier = len(concat)

        op_modules: list[nn.Module] = []
        indices: list[int] = []
        for op_name, idx in ops:
            name = str(op_name).lower().strip()
            if name not in OPS:
                raise ValueError(f"Unknown NAS op: {op_name!r}. Supported: {sorted(OPS)}")
            i = int(idx)
            stride = 2 if self.reduction and i < 2 else 1
            op_modules.append(OPS[name](c_c, stride))
            indices.append(i)

        self.ops = nn.ModuleList(op_modules)
        self.indices = indices

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        s0 = self.pre0(s0)
        s1 = self.pre1(s1)
        states: list[torch.Tensor] = [s0, s1]

        for step in range(self.steps):
            op1 = self.ops[2 * step]
            op2 = self.ops[2 * step + 1]
            i1 = self.indices[2 * step]
            i2 = self.indices[2 * step + 1]

            h1 = op1(states[i1])
            h2 = op2(states[i2])
            if self.training and self.drop_path.p > 0.0:
                h1 = self.drop_path(h1)
                h2 = self.drop_path(h2)
            states.append(h1 + h2)

        return torch.cat([states[i] for i in self.concat], dim=1)


class NASNetworkClassifier(nn.Module):
    """Generic cell-based NAS backbone (DARTS/NASNet/PNAS/ENAS/Amoeba-like)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        genotype: Genotype,
        init_channels: int = 16,
        num_cells: int = 8,
        stem_multiplier: int = 3,
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c0 = max(8, int(round(int(init_channels) * float(width_mult))))
        c_stem = int(stem_multiplier) * c0

        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), c_stem, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c_stem),
            nn.ReLU(inplace=True),
        )

        c_prev_prev = c_stem
        c_prev = c_stem
        c_cur = c0

        n_cells = int(num_cells)
        if n_cells <= 0:
            raise ValueError("num_cells must be > 0")

        reduction_layers = {n_cells // 3, 2 * n_cells // 3}

        cells: list[NASCell] = []
        reduction_prev = False
        for i in range(n_cells):
            reduction = i in reduction_layers
            if reduction:
                c_cur *= 2
            cell = NASCell(
                genotype,
                c_prev_prev=c_prev_prev,
                c_prev=c_prev,
                c_cur=c_cur,
                reduction=reduction,
                reduction_prev=bool(reduction_prev),
                drop_path=float(drop_path),
            )
            cells.append(cell)
            c_prev_prev, c_prev = c_prev, cell.multiplier * c_cur
            reduction_prev = bool(reduction)

        self.cells = nn.ModuleList(cells)
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(c_prev), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = torch.mean(s1, dim=(2, 3))
        out = self.drop(out)
        return self.head(out)
