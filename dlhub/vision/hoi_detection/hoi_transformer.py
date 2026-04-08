from __future__ import annotations
from ._common import build_toy_hoi_detector, smoke_test_hoi
_VARIANTS = {'hoi_transformer_tiny': {'width':24,'depth':1}, 'hoi_transformer_small': {'width':32,'depth':2}, 'hoi_transformer_base': {'width':48,'depth':3}}
def build_hoi_transformer_hoi_detector(*, in_channels:int, num_verbs:int, num_objects:int, variant:str='hoi_transformer_small', width_mult:float=1.0, num_queries:int=16):
    return build_toy_hoi_detector(family='hoi_transformer', variants=_VARIANTS, in_channels=int(in_channels), num_verbs=int(num_verbs), num_objects=int(num_objects), variant=str(variant), width_mult=float(width_mult), num_queries=int(num_queries))
if __name__ == '__main__': smoke_test_hoi(build_hoi_transformer_hoi_detector, 'hoi_transformer_tiny')
