from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'pixelnerf_tiny': {'width':24,'depth':1}, 'pixelnerf_small': {'width':32,'depth':2}, 'pixelnerf_base': {'width':48,'depth':3}}

def build_pixelnerf_view_synthesizer(*, in_channels:int, variant:str='pixelnerf_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='pixelnerf', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_pixelnerf_view_synthesizer, 'pixelnerf_tiny')
