from __future__ import annotations
from ._common import build_toy_spoofer, smoke_test_spoof
_VARIANTS = {'fasformer_tiny': {'width':24,'depth':1}, 'fasformer_small': {'width':32,'depth':2}, 'fasformer_base': {'width':48,'depth':3}}
def build_fasformer_anti_spoofer(*, in_channels:int, variant:str='fasformer_small', width_mult:float=1.0):
    return build_toy_spoofer(family='fasformer', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_spoof(build_fasformer_anti_spoofer, 'fasformer_tiny')
