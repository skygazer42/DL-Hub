from __future__ import annotations
from ._common import build_toy_sketch, smoke_test_sketch
_VARIANTS = {'attention_sketch_tiny': {'width':24,'depth':1,'embed':128}, 'attention_sketch_small': {'width':32,'depth':2,'embed':160}, 'attention_sketch_base': {'width':48,'depth':3,'embed':192}}
def build_attention_sketch_sketch_retriever(*, in_channels:int, variant:str='attention_sketch_small', width_mult:float=1.0):
    return build_toy_sketch(family='attention_sketch', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_sketch(build_attention_sketch_sketch_retriever, 'attention_sketch_tiny')
