from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'sinet_v2_tiny': {'width':24,'depth':1}, 'sinet_v2_small': {'width':32,'depth':2}, 'sinet_v2_base': {'width':48,'depth':3}}
def build_sinet_v2_camouflaged_detector(*, in_channels:int, variant:str='sinet_v2_small', width_mult:float=1.0):
    return build_toy_model(family='sinet_v2', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_model(build_sinet_v2_camouflaged_detector, 'sinet_v2_tiny')
