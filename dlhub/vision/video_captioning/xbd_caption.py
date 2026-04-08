from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'xbd_caption_tiny': {'width':24,'depth':1}, 'xbd_caption_small': {'width':32,'depth':2}, 'xbd_caption_base': {'width':48,'depth':3}}

def build_xbd_caption_video_captioner(*, in_channels:int, variant:str='xbd_caption_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='xbd_caption', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__': smoke_test_model(build_xbd_caption_video_captioner, 'xbd_caption_tiny')
