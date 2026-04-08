from __future__ import annotations
from ._common import build_toy_editor, smoke_test_editor
_VARIANTS = {'mambaedit_tiny': {'width':24,'depth':1}, 'mambaedit_small': {'width':32,'depth':2}, 'mambaedit_base': {'width':48,'depth':3}}
def build_mambaedit_editor(*, in_channels:int, variant:str='mambaedit_small', width_mult:float=1.0):
    return build_toy_editor(family='mambaedit', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_editor(build_mambaedit_editor, 'mambaedit_tiny')
