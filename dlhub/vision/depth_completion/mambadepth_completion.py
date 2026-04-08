from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'mambadepth_completion_tiny': {'width':24,'depth':1}, 'mambadepth_completion_small': {'width':32,'depth':2}, 'mambadepth_completion_base': {'width':48,'depth':3}}

def build_mambadepth_completion_depth_completer(*, in_channels:int, variant:str='mambadepth_completion_small', width_mult:float=1.0):
    return build_toy_model(family='mambadepth_completion', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == '__main__': smoke_test_model(build_mambadepth_completion_depth_completer, 'mambadepth_completion_tiny')
