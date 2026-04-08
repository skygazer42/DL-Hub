from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'spotter_v1_tiny': {'width':24,'depth':1}, 'spotter_v1_small': {'width':32,'depth':2}, 'spotter_v1_base': {'width':48,'depth':3}}

def build_spotter_v1_text_spotter(*, in_channels:int, variant:str='spotter_v1_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='spotter_v1', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__': smoke_test_model(build_spotter_v1_text_spotter, 'spotter_v1_tiny')
