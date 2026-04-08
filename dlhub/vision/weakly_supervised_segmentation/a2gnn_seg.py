from __future__ import annotations
from ._common import build_toy_ws_segmenter, smoke_test_wss
_VARIANTS = {'a2gnn_seg_tiny': {'width':24,'depth':1}, 'a2gnn_seg_small': {'width':32,'depth':2}, 'a2gnn_seg_base': {'width':48,'depth':3}}
def build_a2gnn_seg_ws_segmenter(*, in_channels:int, num_classes:int, variant:str='a2gnn_seg_small', width_mult:float=1.0):
    return build_toy_ws_segmenter(family='a2gnn_seg', variants=_VARIANTS, in_channels=int(in_channels), num_classes=int(num_classes), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_wss(build_a2gnn_seg_ws_segmenter, 'a2gnn_seg_tiny')
