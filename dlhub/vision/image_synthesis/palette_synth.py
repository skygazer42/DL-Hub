from __future__ import annotations
from ._common import build_toy_generator, smoke_test_generator
_VARIANTS = {'palette_synth_tiny': {'width':24,'depth':1}, 'palette_synth_small': {'width':32,'depth':2}, 'palette_synth_base': {'width':48,'depth':3}}
def build_palette_synth_generator(*, in_channels:int, variant:str='palette_synth_small', width_mult:float=1.0):
    return build_toy_generator(family='palette_synth', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_generator(build_palette_synth_generator, 'palette_synth_tiny')
