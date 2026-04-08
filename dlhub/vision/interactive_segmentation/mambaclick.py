from __future__ import annotations
from ._common import build_toy_inter, smoke_test_inter
_VARIANTS = {'mambaclick_tiny': {'width':24,'depth':1}, 'mambaclick_small': {'width':32,'depth':2}, 'mambaclick_base': {'width':48,'depth':3}}

def build_mambaclick_interactive_segmenter(*, in_channels:int, variant:str='mambaclick_small', width_mult:float=1.0):
    return build_toy_inter(family='mambaclick', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == '__main__': smoke_test_inter(build_mambaclick_interactive_segmenter, 'mambaclick_tiny')
