from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'clipseg2_tiny': {'width':24,'depth':1}, 'clipseg2_small': {'width':32,'depth':2}, 'clipseg2_base': {'width':48,'depth':3}}

def build_clipseg2_open_vocab_segmenter(*, in_channels:int, variant:str='clipseg2_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='clipseg2', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_clipseg2_open_vocab_segmenter, 'clipseg2_tiny')
