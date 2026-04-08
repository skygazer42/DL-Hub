from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'visdial_bert_tiny': {'width':24,'depth':1}, 'visdial_bert_small': {'width':32,'depth':2}, 'visdial_bert_base': {'width':48,'depth':3}}

def build_visdial_bert_visual_dialog_model(*, in_channels:int, variant:str='visdial_bert_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='visdial_bert', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__': smoke_test_model(build_visdial_bert_visual_dialog_model, 'visdial_bert_tiny')
