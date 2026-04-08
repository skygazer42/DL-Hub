from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'rainnet_harm_tiny': {'width': 24, 'depth': 1}, 'rainnet_harm_small': {'width': 32, 'depth': 2}, 'rainnet_harm_base': {'width': 48, 'depth': 3}}
def build_rainnet_harm_harmonizer(*, in_channels:int, variant:str='rainnet_harm_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='rainnet_harm', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)
if __name__ == '__main__':
    smoke_test_model(build_rainnet_harm_harmonizer, 'rainnet_harm_tiny')
