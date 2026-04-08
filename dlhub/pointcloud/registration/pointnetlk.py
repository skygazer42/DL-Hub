from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'pointnetlk_tiny': {'width':24,'depth':1}, 'pointnetlk_small': {'width':32,'depth':2}, 'pointnetlk_base': {'width':48,'depth':3}}

def build_pointnetlk_registrar(*, variant:str='pointnetlk_small', width_mult:float=1.0):
    return build_toy_model(family='pointnetlk', variants=_VARIANTS, variant=str(variant), width_mult=float(width_mult))

if __name__ == '__main__': smoke_test_model(build_pointnetlk_registrar, 'pointnetlk_tiny')
