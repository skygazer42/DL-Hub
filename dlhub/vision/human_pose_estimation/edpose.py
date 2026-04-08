from __future__ import annotations
from ._common import build_toy_pose, smoke_test_pose
_VARIANTS = {'edpose_tiny': {'width':24,'depth':1}, 'edpose_small': {'width':32,'depth':2}, 'edpose_base': {'width':48,'depth':3}}
def build_edpose_pose_estimator(*, in_channels:int, num_joints:int, variant:str='edpose_small', width_mult:float=1.0):
    return build_toy_pose(family='edpose', variants=_VARIANTS, in_channels=int(in_channels), num_joints=int(num_joints), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_pose(build_edpose_pose_estimator, 'edpose_tiny')
