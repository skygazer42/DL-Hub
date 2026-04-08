from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'clip4caption_tiny': {'width':24,'depth':1}, 'clip4caption_small': {'width':32,'depth':2}, 'clip4caption_base': {'width':48,'depth':3}}

def build_clip4caption_video_captioner(*, in_channels:int, variant:str='clip4caption_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='clip4caption', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__': smoke_test_model(build_clip4caption_video_captioner, 'clip4caption_tiny')
