from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'cleargrasp_tiny': {'width':24,'depth':1}, 'cleargrasp_small': {'width':32,'depth':2}, 'cleargrasp_base': {'width':48,'depth':3}}

def build_cleargrasp_depth_completer(*, in_channels:int, variant:str='cleargrasp_small', width_mult:float=1.0):
    return build_toy_model(family='cleargrasp', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == '__main__': smoke_test_model(build_cleargrasp_depth_completer, 'cleargrasp_tiny')
