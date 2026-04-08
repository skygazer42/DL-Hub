from __future__ import annotations
from ._common import build_toy_pose6d, smoke_test_6d
_VARIANTS = {'gdrnet_tiny': {'width':24,'depth':1}, 'gdrnet_small': {'width':32,'depth':2}, 'gdrnet_base': {'width':48,'depth':3}}
def build_gdrnet_pose6d_estimator(*, in_channels:int, num_objects:int, variant:str='gdrnet_small', width_mult:float=1.0):
    return build_toy_pose6d(family='gdrnet', variants=_VARIANTS, in_channels=int(in_channels), num_objects=int(num_objects), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_6d(build_gdrnet_pose6d_estimator, 'gdrnet_tiny')
