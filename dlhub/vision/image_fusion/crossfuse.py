from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'crossfuse_tiny': {'width':24,'depth':1}, 'crossfuse_small': {'width':32,'depth':2}, 'crossfuse_base': {'width':48,'depth':3}}
def build_crossfuse_fuser(*, in_channels:int, variant:str='crossfuse_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='crossfuse', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)
if __name__ == '__main__': smoke_test_model(build_crossfuse_fuser, 'crossfuse_tiny')
