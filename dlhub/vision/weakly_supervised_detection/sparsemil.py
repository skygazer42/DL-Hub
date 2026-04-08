from __future__ import annotations
from ._common import build_toy_ws_detector, smoke_test_ws
_VARIANTS = {'sparsemil_tiny': {'width':24,'depth':1}, 'sparsemil_small': {'width':32,'depth':2}, 'sparsemil_base': {'width':48,'depth':3}}
def build_sparsemil_ws_detector(*, in_channels:int, num_classes:int, variant:str='sparsemil_small', width_mult:float=1.0):
    return build_toy_ws_detector(family='sparsemil', variants=_VARIANTS, in_channels=int(in_channels), num_classes=int(num_classes), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_ws(build_sparsemil_ws_detector, 'sparsemil_tiny')
