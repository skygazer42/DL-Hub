from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'uniad_defect_tiny': {'width':24,'depth':1}, 'uniad_defect_small': {'width':32,'depth':2}, 'uniad_defect_base': {'width':48,'depth':3}}

def build_uniad_defect_defect_detector(*, in_channels:int, variant:str='uniad_defect_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='uniad_defect', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_uniad_defect_defect_detector, 'uniad_defect_tiny')
