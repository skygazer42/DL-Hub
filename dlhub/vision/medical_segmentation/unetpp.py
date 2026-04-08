from __future__ import annotations
from torch import nn
from ._common import build_toy_medical_segmenter, smoke_test_med
_VARIANTS = {'unetpp_tiny': {'width':16,'depth':1}, 'unetpp_small': {'width':24,'depth':2}, 'unetpp_base': {'width':32,'depth':3}}
def build_unetpp_medical_segmenter(*, in_channels: int, num_classes: int, variant: str = 'unetpp_small', width_mult: float = 1.0) -> nn.Module:
    return build_toy_medical_segmenter(family='unetpp', variants=_VARIANTS, in_channels=int(in_channels), num_classes=int(num_classes), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__':
    smoke_test_med(build_unetpp_medical_segmenter, 'unetpp_tiny')
