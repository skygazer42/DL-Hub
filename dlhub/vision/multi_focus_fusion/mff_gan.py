from __future__ import annotations
from ._common import build_toy_mff, smoke_test_mff
_VARIANTS = {'mff_gan_tiny': {'width':24,'depth':1}, 'mff_gan_small': {'width':32,'depth':2}, 'mff_gan_base': {'width':48,'depth':3}}
def build_mff_gan_multi_focus_fuser(*, in_channels:int, variant:str='mff_gan_small', width_mult:float=1.0):
    return build_toy_mff(family='mff_gan', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_mff(build_mff_gan_multi_focus_fuser, 'mff_gan_tiny')
