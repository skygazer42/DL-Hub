from __future__ import annotations
from ._common import build_toy_editor, smoke_test_editor
_VARIANTS = {'drag_edit_tiny': {'width':24,'depth':1}, 'drag_edit_small': {'width':32,'depth':2}, 'drag_edit_base': {'width':48,'depth':3}}
def build_drag_edit_editor(*, in_channels:int, variant:str='drag_edit_small', width_mult:float=1.0):
    return build_toy_editor(family='drag_edit', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_editor(build_drag_edit_editor, 'drag_edit_tiny')
