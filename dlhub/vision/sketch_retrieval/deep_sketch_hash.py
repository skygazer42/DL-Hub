from __future__ import annotations
from ._common import build_toy_sketch, smoke_test_sketch
_VARIANTS = {'deep_sketch_hash_tiny': {'width':24,'depth':1,'embed':128}, 'deep_sketch_hash_small': {'width':32,'depth':2,'embed':160}, 'deep_sketch_hash_base': {'width':48,'depth':3,'embed':192}}
def build_deep_sketch_hash_sketch_retriever(*, in_channels:int, variant:str='deep_sketch_hash_small', width_mult:float=1.0):
    return build_toy_sketch(family='deep_sketch_hash', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_sketch(build_deep_sketch_hash_sketch_retriever, 'deep_sketch_hash_tiny')
