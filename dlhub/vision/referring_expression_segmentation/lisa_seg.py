from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'lisa_seg_tiny': {'width':24,'depth':1}, 'lisa_seg_small': {'width':32,'depth':2}, 'lisa_seg_base': {'width':48,'depth':3}}

def build_lisa_seg_refexp_segmenter(*, in_channels:int, variant:str='lisa_seg_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='lisa_seg', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_lisa_seg_refexp_segmenter, 'lisa_seg_tiny')
