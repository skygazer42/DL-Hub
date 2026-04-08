from __future__ import annotations
from ._common import build_toy_text_detector, smoke_test_text
_VARIANTS = {'textfusenet_tiny': {'width':24,'depth':1}, 'textfusenet_small': {'width':32,'depth':2}, 'textfusenet_base': {'width':48,'depth':3}}
def build_textfusenet_text_detector(*, in_channels:int, variant:str='textfusenet_small', width_mult:float=1.0):
    return build_toy_text_detector(family='textfusenet', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_text(build_textfusenet_text_detector, 'textfusenet_tiny')
