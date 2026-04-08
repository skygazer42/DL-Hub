from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'s2am_tiny': {'width': 24, 'depth': 1}, 's2am_small': {'width': 32, 'depth': 2}, 's2am_base': {'width': 48, 'depth': 3}}
def build_s2am_harmonizer(*, in_channels:int, variant:str='s2am_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='s2am', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)
if __name__ == '__main__':
    smoke_test_model(build_s2am_harmonizer, 's2am_tiny')
