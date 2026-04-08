from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'tapg_tiny': {'width':24,'depth':1}, 'tapg_small': {'width':32,'depth':2}, 'tapg_base': {'width':48,'depth':3}}
def build_tapg_tal_model(*, in_channels:int, variant:str='tapg_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='tapg', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)
if __name__ == '__main__': smoke_test_model(build_tapg_tal_model, 'tapg_tiny')
