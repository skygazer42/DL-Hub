from __future__ import annotations
from ._common import build_toy_hw, smoke_test_hw
_VARIANTS = {'transformer_hwr_tiny': {'width':24,'depth':1}, 'transformer_hwr_small': {'width':32,'depth':2}, 'transformer_hwr_base': {'width':48,'depth':3}}
def build_transformer_hwr_handwriting_recognizer(*, input_dim:int, vocab_size:int, variant:str='transformer_hwr_small', width_mult:float=1.0):
    return build_toy_hw(family='transformer_hwr', variants=_VARIANTS, input_dim=int(input_dim), vocab_size=int(vocab_size), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_hw(build_transformer_hwr_handwriting_recognizer, 'transformer_hwr_tiny')
