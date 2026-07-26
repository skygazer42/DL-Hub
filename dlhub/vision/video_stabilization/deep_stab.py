from __future__ import annotations
from ._common import build_toy_stabilizer, smoke_test_stabilizer
_VARIANTS = {'deep_stab_tiny': {'width':24,'depth':1}, 'deep_stab_small': {'width':32,'depth':2}, 'deep_stab_base': {'width':48,'depth':3}}
def build_deep_stab_stabilizer(*, in_channels:int, variant:str='deep_stab_small', width_mult:float=1.0):
    return build_toy_stabilizer(family='deep_stab', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_stabilizer(build_deep_stab_stabilizer, 'deep_stab_tiny')
