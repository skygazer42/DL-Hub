from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'gaze360_tiny': {'width':24,'depth':1}, 'gaze360_small': {'width':32,'depth':2}, 'gaze360_base': {'width':48,'depth':3}}
def build_gaze360_gaze_estimator(*, in_channels:int, variant:str='gaze360_small', width_mult:float=1.0):
    return build_toy_model(family='gaze360', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_model(build_gaze360_gaze_estimator, 'gaze360_tiny')
