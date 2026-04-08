from __future__ import annotations
from ._common import build_toy_video_enhancer, smoke_test_ve
_VARIANTS = {'videoclean_tiny': {'width':24,'depth':1}, 'videoclean_small': {'width':32,'depth':2}, 'videoclean_base': {'width':48,'depth':3}}
def build_videoclean_video_enhancer(*, in_channels:int, variant:str='videoclean_small', width_mult:float=1.0):
    return build_toy_video_enhancer(family='videoclean', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_ve(build_videoclean_video_enhancer, 'videoclean_tiny')
