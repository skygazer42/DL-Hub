from __future__ import annotations
from ._common import build_toy_vos, smoke_test_vos
_VARIANTS = {'savos_tiny': {'width':24,'depth':1}, 'savos_small': {'width':32,'depth':2}, 'savos_base': {'width':48,'depth':3}}
def build_savos_vos_model(*, in_channels:int, variant:str='savos_small', width_mult:float=1.0, num_masks:int=2):
    return build_toy_vos(family='savos', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), num_masks=int(num_masks))
if __name__ == '__main__': smoke_test_vos(build_savos_vos_model, 'savos_tiny')
