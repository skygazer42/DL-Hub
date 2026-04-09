from __future__ import annotations
from ._common import build_toy_counter, smoke_test_counter
_VARIANTS = {'mambacount_obj_tiny': {'width':24,'depth':1}, 'mambacount_obj_small': {'width':32,'depth':2}, 'mambacount_obj_base': {'width':48,'depth':3}}
def build_mambacount_obj_(*, in_channels:int, variant:str='mambacount_obj_small', width_mult:float=1.0):
    return build_toy_counter(family='mambacount_obj', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_counter(build_mambacount_obj_, 'mambacount_obj_tiny')