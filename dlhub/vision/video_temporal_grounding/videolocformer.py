from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'videolocformer_tiny': {'width':24,'depth':1}, 'videolocformer_small': {'width':32,'depth':2}, 'videolocformer_base': {'width':48,'depth':3}}

def build_videolocformer_temporal_grounder(*, in_channels:int, variant:str='videolocformer_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='videolocformer', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_videolocformer_temporal_grounder, 'videolocformer_tiny')
