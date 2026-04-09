from __future__ import annotations
from ._common import build_toy_prompt_learner, smoke_test_prompt_learner
_VARIANTS = {'prefix_fusion_prompt_tiny': {'width':24,'depth':1}, 'prefix_fusion_prompt_small': {'width':32,'depth':2}, 'prefix_fusion_prompt_base': {'width':48,'depth':3}}
def build_prefix_fusion_prompt_prompt_learner(*, in_channels:int, variant:str='prefix_fusion_prompt_small', width_mult:float=1.0, prompt_len:int=8):
    return build_toy_prompt_learner(family='prefix_fusion_prompt', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), prompt_len=int(prompt_len))
if __name__ == '__main__': smoke_test_prompt_learner(build_prefix_fusion_prompt_prompt_learner, 'prefix_fusion_prompt_tiny')
