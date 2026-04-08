from __future__ import annotations
from ._common import build_toy_change, smoke_test_change
_VARIANTS = {'detchange_tiny': {'width':24,'depth':1}, 'detchange_small': {'width':32,'depth':2}, 'detchange_base': {'width':48,'depth':3}}
def build_detchange_change_detector(*, in_channels:int, variant:str='detchange_small', width_mult:float=1.0):
    return build_toy_change(family='detchange', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_change(build_detchange_change_detector, 'detchange_tiny')
