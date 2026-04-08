from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'dust3r_match_tiny': {'width':24,'depth':1,'embed':128}, 'dust3r_match_small': {'width':32,'depth':2,'embed':160}, 'dust3r_match_base': {'width':48,'depth':3,'embed':192}}
def build_dust3r_match_image_matcher(*, in_channels:int, variant:str='dust3r_match_small', width_mult:float=1.0):
    return build_toy_model(family='dust3r_match', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_model(build_dust3r_match_image_matcher, 'dust3r_match_tiny')
