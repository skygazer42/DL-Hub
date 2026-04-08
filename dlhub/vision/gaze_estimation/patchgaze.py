from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'patchgaze_tiny': {'width':24,'depth':1}, 'patchgaze_small': {'width':32,'depth':2}, 'patchgaze_base': {'width':48,'depth':3}}
def build_patchgaze_gaze_estimator(*, in_channels:int, variant:str='patchgaze_small', width_mult:float=1.0):
    return build_toy_model(family='patchgaze', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_model(build_patchgaze_gaze_estimator, 'patchgaze_tiny')
