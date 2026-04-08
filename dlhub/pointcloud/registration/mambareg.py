from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'mambareg_tiny': {'width':24,'depth':1}, 'mambareg_small': {'width':32,'depth':2}, 'mambareg_base': {'width':48,'depth':3}}

def build_mambareg_registrar(*, variant:str='mambareg_small', width_mult:float=1.0):
    return build_toy_model(family='mambareg', variants=_VARIANTS, variant=str(variant), width_mult=float(width_mult))

if __name__ == '__main__': smoke_test_model(build_mambareg_registrar, 'mambareg_tiny')
