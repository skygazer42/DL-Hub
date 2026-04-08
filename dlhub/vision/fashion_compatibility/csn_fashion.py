from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'csn_fashion_tiny': {'width':24,'depth':1}, 'csn_fashion_small': {'width':32,'depth':2}, 'csn_fashion_base': {'width':48,'depth':3}}

def build_csn_fashion_fashion_compat_model(*, in_channels:int, variant:str='csn_fashion_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='csn_fashion', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_csn_fashion_fashion_compat_model, 'csn_fashion_tiny')
