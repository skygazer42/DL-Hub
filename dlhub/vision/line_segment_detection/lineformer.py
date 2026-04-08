from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'lineformer_tiny': {'width':24,'depth':1}, 'lineformer_small': {'width':32,'depth':2}, 'lineformer_base': {'width':48,'depth':3}}

def build_lineformer_line_detector(*, in_channels:int, variant:str='lineformer_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='lineformer', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_lineformer_line_detector, 'lineformer_tiny')
