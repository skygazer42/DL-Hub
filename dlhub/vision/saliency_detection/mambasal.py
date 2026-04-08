from __future__ import annotations
from torch import nn
from ._common import build_toy_restoration, smoke_test_restoration
_VARIANTS = {'mambasal_tiny': {'width': 24, 'depth': 1}, 'mambasal_small': {'width': 32, 'depth': 2}, 'mambasal_base': {'width': 48, 'depth': 3}}
def build_mambasal_saliency_detector(*, in_channels: int, variant: str = 'mambasal_small', width_mult: float = 1.0) -> nn.Module:
    return build_toy_restoration(family='mambasal', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), out_key='saliency')
if __name__ == '__main__':
    smoke_test_restoration(build_mambasal_saliency_detector, 'mambasal_tiny', 'saliency')
