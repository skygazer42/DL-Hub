from __future__ import annotations
from ._common import build_toy_editor, smoke_test_editor
_VARIANTS = {'prompt2prompt_tiny': {'width':24,'depth':1}, 'prompt2prompt_small': {'width':32,'depth':2}, 'prompt2prompt_base': {'width':48,'depth':3}}
def build_prompt2prompt_editor(*, in_channels:int, variant:str='prompt2prompt_small', width_mult:float=1.0):
    return build_toy_editor(family='prompt2prompt', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_editor(build_prompt2prompt_editor, 'prompt2prompt_tiny')
