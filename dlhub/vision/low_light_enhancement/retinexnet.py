from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'retinexnet_tiny': {'width': 24, 'depth': 1}, 'retinexnet_small': {'width': 32, 'depth': 2}, 'retinexnet_base': {'width': 48, 'depth': 3}}

def build_retinexnet_enhancer(*, in_channels:int, variant:str='retinexnet_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='retinexnet', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_retinexnet_enhancer, 'retinexnet_tiny')
