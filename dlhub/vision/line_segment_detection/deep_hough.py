from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'deep_hough_tiny': {'width':24,'depth':1}, 'deep_hough_small': {'width':32,'depth':2}, 'deep_hough_base': {'width':48,'depth':3}}

def build_deep_hough_line_detector(*, in_channels:int, variant:str='deep_hough_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='deep_hough', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_deep_hough_line_detector, 'deep_hough_tiny')
