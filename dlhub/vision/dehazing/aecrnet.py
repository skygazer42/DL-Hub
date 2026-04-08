from __future__ import annotations
from torch import nn
from ._common import build_toy_restoration, smoke_test_restoration
_VARIANTS = {'aecrnet_tiny': {'width': 24, 'depth': 1}, 'aecrnet_small': {'width': 32, 'depth': 2}, 'aecrnet_base': {'width': 48, 'depth': 3}}
def build_aecrnet_dehazer(*, in_channels: int, variant: str = 'aecrnet_small', width_mult: float = 1.0) -> nn.Module:
    return build_toy_restoration(family='aecrnet', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), out_key='dehazed')
if __name__ == '__main__':
    smoke_test_restoration(build_aecrnet_dehazer, 'aecrnet_tiny', 'dehazed')
