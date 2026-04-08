from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'mambasymbol_tiny': {'width':24,'depth':1}, 'mambasymbol_small': {'width':32,'depth':2}, 'mambasymbol_base': {'width':48,'depth':3}}

def build_mambasymbol_symbol_recognizer(*, in_channels:int, variant:str='mambasymbol_small', width_mult:float=1.0, **kwargs):
    return build_toy_model(family='mambasymbol', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), **kwargs)

if __name__ == '__main__':
    smoke_test_model(build_mambasymbol_symbol_recognizer, 'mambasymbol_tiny')
