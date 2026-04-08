from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'layoutlm_layout_tiny': {'width':24,'depth':1}, 'layoutlm_layout_small': {'width':32,'depth':2}, 'layoutlm_layout_base': {'width':48,'depth':3}}

def build_layoutlm_layout_layout_analyzer(*, in_channels:int, variant:str='layoutlm_layout_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='layoutlm_layout', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_layoutlm_layout_layout_analyzer, 'layoutlm_layout_tiny')
