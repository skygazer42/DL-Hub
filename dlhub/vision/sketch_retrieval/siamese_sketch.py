from __future__ import annotations
from ._common import build_toy_sketch, smoke_test_sketch
_VARIANTS = {'siamese_sketch_tiny': {'width':24,'depth':1,'embed':128}, 'siamese_sketch_small': {'width':32,'depth':2,'embed':160}, 'siamese_sketch_base': {'width':48,'depth':3,'embed':192}}
def build_siamese_sketch_sketch_retriever(*, in_channels:int, variant:str='siamese_sketch_small', width_mult:float=1.0):
    return build_toy_sketch(family='siamese_sketch', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_sketch(build_siamese_sketch_sketch_retriever, 'siamese_sketch_tiny')
