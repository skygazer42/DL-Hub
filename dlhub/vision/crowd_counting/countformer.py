from __future__ import annotations
from ._common import build_toy_counter, smoke_test_counter
_VARIANTS = {'countformer_tiny': {'width':24,'depth':1}, 'countformer_small': {'width':32,'depth':2}, 'countformer_base': {'width':48,'depth':3}}
def build_countformer_crowd_counter(*, in_channels:int, variant:str='countformer_small', width_mult:float=1.0):
    return build_toy_counter(family='countformer', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_counter(build_countformer_crowd_counter, 'countformer_tiny')
