from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'vdtrans_tiny': {'width':24,'depth':1}, 'vdtrans_small': {'width':32,'depth':2}, 'vdtrans_base': {'width':48,'depth':3}}

def build_vdtrans_visual_dialog_model(*, in_channels:int, variant:str='vdtrans_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='vdtrans', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__': smoke_test_model(build_vdtrans_visual_dialog_model, 'vdtrans_tiny')
