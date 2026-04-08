from __future__ import annotations
from torch import nn
from ._common import build_toy_ocr_model, smoke_test_ocr

_VARIANTS = {
    'parseq_tiny': {'width': 32, 'depth': 1},
    'parseq_small': {'width': 48, 'depth': 2},
    'parseq_base': {'width': 64, 'depth': 3},
}

def build_parseq_ocr_model(*, in_channels: int, vocab_size: int, seq_len: int = 16, variant: str = 'parseq_small', width_mult: float = 1.0) -> nn.Module:
    return build_toy_ocr_model(family='parseq', variants=_VARIANTS, in_channels=int(in_channels), vocab_size=int(vocab_size), seq_len=int(seq_len), variant=str(variant), width_mult=float(width_mult), decoder_mode='transformer')

if __name__ == '__main__':
    smoke_test_ocr(build_parseq_ocr_model, 'parseq_tiny')
