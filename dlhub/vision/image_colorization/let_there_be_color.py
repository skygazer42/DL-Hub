from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'let_there_be_color_tiny': {'width': 24, 'depth': 1}, 'let_there_be_color_small': {'width': 32, 'depth': 2}, 'let_there_be_color_base': {'width': 48, 'depth': 3}}

def build_let_there_be_color_colorizer(*, in_channels:int, variant:str='let_there_be_color_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='let_there_be_color', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_let_there_be_color_colorizer, 'let_there_be_color_tiny')
