from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'oriented_spot_tiny': {'width':24,'depth':1}, 'oriented_spot_small': {'width':32,'depth':2}, 'oriented_spot_base': {'width':48,'depth':3}}

def build_oriented_spot_text_spotter(*, in_channels:int, variant:str='oriented_spot_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='oriented_spot', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__': smoke_test_model(build_oriented_spot_text_spotter, 'oriented_spot_tiny')
