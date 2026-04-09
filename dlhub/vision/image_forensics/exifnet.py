from __future__ import annotations
from torch import nn
from ._common import build_toy_anomaly_detector, smoke_test_anomaly
_VARIANTS = {'exifnet_tiny': {'width':24,'depth':1}, 'exifnet_small': {'width':32,'depth':2}, 'exifnet_base': {'width':48,'depth':3}}
def build_exifnet_(*, in_channels: int, variant: str = 'exifnet_small', width_mult: float = 1.0) -> nn.Module:
    return build_toy_anomaly_detector(family='exifnet', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__':
    smoke_test_anomaly(build_exifnet_, 'exifnet_tiny')