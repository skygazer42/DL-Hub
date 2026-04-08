from __future__ import annotations
from ._common import build_toy_video_restorer, smoke_test_video
_VARIANTS = {'rvrt_tiny': {'width':24,'depth':1}, 'rvrt_small': {'width':32,'depth':2}, 'rvrt_base': {'width':48,'depth':3}}
def build_rvrt_video_restorer(*, in_channels:int, variant:str='rvrt_small', width_mult:float=1.0):
    return build_toy_video_restorer(family='rvrt', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_video(build_rvrt_video_restorer, 'rvrt_tiny')
