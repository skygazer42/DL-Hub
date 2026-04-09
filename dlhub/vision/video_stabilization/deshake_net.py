from __future__ import annotations
from ._common import build_toy_stabilizer, smoke_test_stabilizer
_VARIANTS = {'deshake_net_tiny': {'width':24,'depth':1}, 'deshake_net_small': {'width':32,'depth':2}, 'deshake_net_base': {'width':48,'depth':3}}
def build_deshake_net_stabilizer(*, in_channels:int, variant:str='deshake_net_small', width_mult:float=1.0):
    return build_toy_stabilizer(family='deshake_net', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_stabilizer(build_deshake_net_stabilizer, 'deshake_net_tiny')
