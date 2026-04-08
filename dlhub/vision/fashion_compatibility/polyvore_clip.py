from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'polyvore_clip_tiny': {'width':24,'depth':1}, 'polyvore_clip_small': {'width':32,'depth':2}, 'polyvore_clip_base': {'width':48,'depth':3}}

def build_polyvore_clip_fashion_compat_model(*, in_channels:int, variant:str='polyvore_clip_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='polyvore_clip', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_polyvore_clip_fashion_compat_model, 'polyvore_clip_tiny')
