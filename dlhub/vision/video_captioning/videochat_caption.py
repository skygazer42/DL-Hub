from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'videochat_caption_tiny': {'width':24,'depth':1}, 'videochat_caption_small': {'width':32,'depth':2}, 'videochat_caption_base': {'width':48,'depth':3}}

def build_videochat_caption_video_captioner(*, in_channels:int, variant:str='videochat_caption_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='videochat_caption', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__': smoke_test_model(build_videochat_caption_video_captioner, 'videochat_caption_tiny')
