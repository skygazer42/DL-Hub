from __future__ import annotations
from ._common import build_toy_vis, smoke_test_vis
_VARIANTS = {'mambavis_tiny': {'width':24,'depth':1}, 'mambavis_small': {'width':32,'depth':2}, 'mambavis_base': {'width':48,'depth':3}}
def build_mambavis_video_instance_segmenter(*, in_channels:int, variant:str='mambavis_small', width_mult:float=1.0, num_instances:int=8):
    return build_toy_vis(family='mambavis', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), num_instances=int(num_instances))
if __name__ == '__main__': smoke_test_vis(build_mambavis_video_instance_segmenter, 'mambavis_tiny')
