from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'nodeedge_doc_tiny': {'width':24,'depth':1}, 'nodeedge_doc_small': {'width':32,'depth':2}, 'nodeedge_doc_base': {'width':48,'depth':3}}

def build_nodeedge_doc_(*, in_channels:int, variant:str='nodeedge_doc_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='nodeedge_doc', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_nodeedge_doc_, 'nodeedge_doc_tiny')