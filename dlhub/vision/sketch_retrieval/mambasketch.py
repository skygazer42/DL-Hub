from __future__ import annotations
from ._common import build_toy_sketch, smoke_test_sketch
_VARIANTS = {'mambasketch_tiny': {'width':24,'depth':1,'embed':128}, 'mambasketch_small': {'width':32,'depth':2,'embed':160}, 'mambasketch_base': {'width':48,'depth':3,'embed':192}}
def build_mambasketch_sketch_retriever(*, in_channels:int, variant:str='mambasketch_small', width_mult:float=1.0):
    return build_toy_sketch(family='mambasketch', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_sketch(build_mambasketch_sketch_retriever, 'mambasketch_tiny')
