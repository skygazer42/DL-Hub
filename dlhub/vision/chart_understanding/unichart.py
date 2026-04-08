from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'unichart_tiny': {'width':24,'depth':1}, 'unichart_small': {'width':32,'depth':2}, 'unichart_base': {'width':48,'depth':3}}

def build_unichart_chart_understander(*, in_channels:int, variant:str='unichart_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='unichart', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_unichart_chart_understander, 'unichart_tiny')
