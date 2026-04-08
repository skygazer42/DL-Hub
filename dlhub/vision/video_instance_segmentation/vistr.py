from __future__ import annotations
from ._common import build_toy_vis, smoke_test_vis
_VARIANTS = {'vistr_tiny': {'width':24,'depth':1}, 'vistr_small': {'width':32,'depth':2}, 'vistr_base': {'width':48,'depth':3}}
def build_vistr_video_instance_segmenter(*, in_channels:int, variant:str='vistr_small', width_mult:float=1.0, num_instances:int=8):
    return build_toy_vis(family='vistr', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), num_instances=int(num_instances))
if __name__ == '__main__': smoke_test_vis(build_vistr_video_instance_segmenter, 'vistr_tiny')
