from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'neuralangelo2d_tiny': {'width':24,'depth':1}, 'neuralangelo2d_small': {'width':32,'depth':2}, 'neuralangelo2d_base': {'width':48,'depth':3}}

def build_neuralangelo2d_view_synthesizer(*, in_channels:int, variant:str='neuralangelo2d_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='neuralangelo2d', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_neuralangelo2d_view_synthesizer, 'neuralangelo2d_tiny')
