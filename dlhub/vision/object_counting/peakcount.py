from __future__ import annotations
from ._common import build_toy_counter, smoke_test_counter
_VARIANTS = {'peakcount_tiny': {'width':24,'depth':1}, 'peakcount_small': {'width':32,'depth':2}, 'peakcount_base': {'width':48,'depth':3}}
def build_peakcount_(*, in_channels:int, variant:str='peakcount_small', width_mult:float=1.0):
    return build_toy_counter(family='peakcount', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_counter(build_peakcount_, 'peakcount_tiny')