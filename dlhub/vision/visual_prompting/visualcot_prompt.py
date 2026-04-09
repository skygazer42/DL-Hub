from __future__ import annotations
from ._common import build_toy_inter, smoke_test_inter
_VARIANTS = {'visualcot_prompt_tiny': {'width':24,'depth':1}, 'visualcot_prompt_small': {'width':32,'depth':2}, 'visualcot_prompt_base': {'width':48,'depth':3}}

def build_visualcot_prompt_(*, in_channels:int, variant:str='visualcot_prompt_small', width_mult:float=1.0):
    return build_toy_inter(family='visualcot_prompt', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == '__main__': smoke_test_inter(build_visualcot_prompt_, 'visualcot_prompt_tiny')