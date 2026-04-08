from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'hit_cod_tiny': {'width':24,'depth':1}, 'hit_cod_small': {'width':32,'depth':2}, 'hit_cod_base': {'width':48,'depth':3}}
def build_hit_cod_camouflaged_detector(*, in_channels:int, variant:str='hit_cod_small', width_mult:float=1.0):
    return build_toy_model(family='hit_cod', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_model(build_hit_cod_camouflaged_detector, 'hit_cod_tiny')
