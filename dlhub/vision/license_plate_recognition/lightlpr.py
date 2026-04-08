from __future__ import annotations
from ._common import build_toy_plate, smoke_test_plate
_VARIANTS = {'lightlpr_tiny': {'width':24,'depth':1}, 'lightlpr_small': {'width':32,'depth':2}, 'lightlpr_base': {'width':48,'depth':3}}
def build_lightlpr_plate_recognizer(*, in_channels:int, vocab_size:int, seq_len:int=10, variant:str='lightlpr_small', width_mult:float=1.0):
    return build_toy_plate(family='lightlpr', variants=_VARIANTS, in_channels=int(in_channels), vocab_size=int(vocab_size), seq_len=int(seq_len), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_plate(build_lightlpr_plate_recognizer, 'lightlpr_tiny')
