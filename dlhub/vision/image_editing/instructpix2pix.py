from __future__ import annotations
from ._common import build_toy_editor, smoke_test_editor
_VARIANTS = {'instructpix2pix_tiny': {'width':24,'depth':1}, 'instructpix2pix_small': {'width':32,'depth':2}, 'instructpix2pix_base': {'width':48,'depth':3}}
def build_instructpix2pix_editor(*, in_channels:int, variant:str='instructpix2pix_small', width_mult:float=1.0):
    return build_toy_editor(family='instructpix2pix', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_editor(build_instructpix2pix_editor, 'instructpix2pix_tiny')
