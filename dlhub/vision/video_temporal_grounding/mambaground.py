from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'mambaground_tiny': {'width':24,'depth':1}, 'mambaground_small': {'width':32,'depth':2}, 'mambaground_base': {'width':48,'depth':3}}

def build_mambaground_temporal_grounder(*, in_channels:int, variant:str='mambaground_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='mambaground', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_mambaground_temporal_grounder, 'mambaground_tiny')
