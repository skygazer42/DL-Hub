from __future__ import annotations
from torch import nn
from ._common import build_toy_retrieval_model, smoke_test_retrieval
_VARIANTS = {'regional_tiny': {'width':24,'depth':1,'embed':128}, 'regional_small': {'width':32,'depth':2,'embed':160}, 'regional_base': {'width':48,'depth':3,'embed':192}}
def build_regional_retrieval_model(*, in_channels: int, variant: str = 'regional_small', width_mult: float = 1.0) -> nn.Module:
    return build_toy_retrieval_model(family='regional', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__':
    smoke_test_retrieval(build_regional_retrieval_model, 'regional_tiny')
