from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'d2sa_tiny': {'width':24,'depth':1}, 'd2sa_small': {'width':32,'depth':2}, 'd2sa_base': {'width':48,'depth':3}}

def build_d2sa_edge_detector(*, in_channels:int, variant:str='d2sa_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='d2sa', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_d2sa_edge_detector, 'd2sa_tiny')
