from __future__ import annotations
from ._common import build_toy_pose, smoke_test_pose
_VARIANTS = {'keytrack_tiny': {'width':24,'depth':1}, 'keytrack_small': {'width':32,'depth':2}, 'keytrack_base': {'width':48,'depth':3}}
def build_keytrack_(*, in_channels:int, num_joints:int, variant:str='keytrack_small', width_mult:float=1.0):
    return build_toy_pose(family='keytrack', variants=_VARIANTS, in_channels=int(in_channels), num_joints=int(num_joints), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_pose(build_keytrack_, 'keytrack_tiny')