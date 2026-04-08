from __future__ import annotations
from ._common import build_toy_ws_segmenter, smoke_test_wss
_VARIANTS = {'seam_tiny': {'width':24,'depth':1}, 'seam_small': {'width':32,'depth':2}, 'seam_base': {'width':48,'depth':3}}
def build_seam_ws_segmenter(*, in_channels:int, num_classes:int, variant:str='seam_small', width_mult:float=1.0):
    return build_toy_ws_segmenter(family='seam', variants=_VARIANTS, in_channels=int(in_channels), num_classes=int(num_classes), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_wss(build_seam_ws_segmenter, 'seam_tiny')
