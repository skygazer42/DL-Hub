from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'outfit_transformer_tiny': {'width':24,'depth':1}, 'outfit_transformer_small': {'width':32,'depth':2}, 'outfit_transformer_base': {'width':48,'depth':3}}

def build_outfit_transformer_fashion_compat_model(*, in_channels:int, variant:str='outfit_transformer_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='outfit_transformer', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_outfit_transformer_fashion_compat_model, 'outfit_transformer_tiny')
