from __future__ import annotations
from torch import nn
from ._common import build_toy_restoration, smoke_test_restoration
_VARIANTS = {'ffanet_tiny': {'width': 24, 'depth': 1}, 'ffanet_small': {'width': 32, 'depth': 2}, 'ffanet_base': {'width': 48, 'depth': 3}}
def build_ffanet_dehazer(*, in_channels: int, variant: str = 'ffanet_small', width_mult: float = 1.0) -> nn.Module:
    return build_toy_restoration(family='ffanet', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), out_key='dehazed')
if __name__ == '__main__':
    smoke_test_restoration(build_ffanet_dehazer, 'ffanet_tiny', 'dehazed')
