from __future__ import annotations
from torch import nn
from ._common import build_toy_rs_detector, smoke_test_rs
_VARIANTS = {'retinanet_tiny': {'width':24,'depth':1}, 'retinanet_small': {'width':32,'depth':2}, 'retinanet_base': {'width':48,'depth':3}}
def build_retinanet_rs_detector(*, in_channels: int, num_classes: int, variant: str = 'retinanet_small', width_mult: float = 1.0) -> nn.Module:
    return build_toy_rs_detector(family='retinanet', variants=_VARIANTS, in_channels=int(in_channels), num_classes=int(num_classes), variant=str(variant), width_mult=float(width_mult), oriented=True)
if __name__ == '__main__':
    smoke_test_rs(build_retinanet_rs_detector, 'retinanet_tiny')
