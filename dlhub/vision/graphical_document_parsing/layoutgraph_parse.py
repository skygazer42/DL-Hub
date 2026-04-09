from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'layoutgraph_parse_tiny': {'width':24,'depth':1}, 'layoutgraph_parse_small': {'width':32,'depth':2}, 'layoutgraph_parse_base': {'width':48,'depth':3}}

def build_layoutgraph_parse_(*, in_channels:int, variant:str='layoutgraph_parse_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='layoutgraph_parse', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_layoutgraph_parse_, 'layoutgraph_parse_tiny')