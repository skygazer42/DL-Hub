from __future__ import annotations
from torch import nn
from ._common import build_toy_anomaly_detector, smoke_test_anomaly
_VARIANTS = {'tracecnn_tiny': {'width':24,'depth':1}, 'tracecnn_small': {'width':32,'depth':2}, 'tracecnn_base': {'width':48,'depth':3}}
def build_tracecnn_(*, in_channels: int, variant: str = 'tracecnn_small', width_mult: float = 1.0) -> nn.Module:
    return build_toy_anomaly_detector(family='tracecnn', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__':
    smoke_test_anomaly(build_tracecnn_, 'tracecnn_tiny')