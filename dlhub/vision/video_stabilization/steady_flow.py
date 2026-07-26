from __future__ import annotations
from ._common import build_toy_stabilizer, smoke_test_stabilizer
_VARIANTS = {'steady_flow_tiny': {'width':24,'depth':1}, 'steady_flow_small': {'width':32,'depth':2}, 'steady_flow_base': {'width':48,'depth':3}}
def build_steady_flow_stabilizer(*, in_channels:int, variant:str='steady_flow_small', width_mult:float=1.0):
    return build_toy_stabilizer(family='steady_flow', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_stabilizer(build_steady_flow_stabilizer, 'steady_flow_tiny')
