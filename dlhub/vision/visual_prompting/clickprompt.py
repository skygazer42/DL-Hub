from __future__ import annotations
from ._common import build_toy_inter, smoke_test_inter
_VARIANTS = {'clickprompt_tiny': {'width':24,'depth':1}, 'clickprompt_small': {'width':32,'depth':2}, 'clickprompt_base': {'width':48,'depth':3}}

def build_clickprompt_(*, in_channels:int, variant:str='clickprompt_small', width_mult:float=1.0):
    return build_toy_inter(family='clickprompt', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))

if __name__ == '__main__': smoke_test_inter(build_clickprompt_, 'clickprompt_tiny')