from __future__ import annotations
from ._common import build_toy_attr, smoke_test_attr
_VARIANTS = {'multitask_pedattr_tiny': {'width':24,'depth':1}, 'multitask_pedattr_small': {'width':32,'depth':2}, 'multitask_pedattr_base': {'width':48,'depth':3}}
def build_multitask_pedattr_(*, in_channels:int, num_attributes:int, variant:str='multitask_pedattr_small', width_mult:float=1.0):
    return build_toy_attr(family='multitask_pedattr', variants=_VARIANTS, in_channels=int(in_channels), num_attributes=int(num_attributes), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_attr(build_multitask_pedattr_, 'multitask_pedattr_tiny')