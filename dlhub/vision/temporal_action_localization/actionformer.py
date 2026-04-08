from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'actionformer_tiny': {'width':24,'depth':1}, 'actionformer_small': {'width':32,'depth':2}, 'actionformer_base': {'width':48,'depth':3}}
def build_actionformer_tal_model(*, in_channels:int, variant:str='actionformer_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='actionformer', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)
if __name__ == '__main__': smoke_test_model(build_actionformer_tal_model, 'actionformer_tiny')
