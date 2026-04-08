from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'reverse_defect_tiny': {'width':24,'depth':1}, 'reverse_defect_small': {'width':32,'depth':2}, 'reverse_defect_base': {'width':48,'depth':3}}

def build_reverse_defect_defect_detector(*, in_channels:int, variant:str='reverse_defect_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='reverse_defect', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_reverse_defect_defect_detector, 'reverse_defect_tiny')
