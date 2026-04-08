from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'docdet_layout_tiny': {'width':24,'depth':1}, 'docdet_layout_small': {'width':32,'depth':2}, 'docdet_layout_base': {'width':48,'depth':3}}

def build_docdet_layout_layout_analyzer(*, in_channels:int, variant:str='docdet_layout_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='docdet_layout', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_docdet_layout_layout_analyzer, 'docdet_layout_tiny')
