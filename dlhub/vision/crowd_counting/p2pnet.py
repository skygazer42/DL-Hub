from __future__ import annotations
from ._common import build_toy_counter, smoke_test_counter
_VARIANTS = {'p2pnet_tiny': {'width':24,'depth':1}, 'p2pnet_small': {'width':32,'depth':2}, 'p2pnet_base': {'width':48,'depth':3}}
def build_p2pnet_crowd_counter(*, in_channels:int, variant:str='p2pnet_small', width_mult:float=1.0):
    return build_toy_counter(family='p2pnet', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_counter(build_p2pnet_crowd_counter, 'p2pnet_tiny')
