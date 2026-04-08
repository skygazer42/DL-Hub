from __future__ import annotations
from torch import nn
from ._common import build_toy_ocr_model, smoke_test_ocr

_VARIANTS = {
    'sar_tiny': {'width': 32, 'depth': 1},
    'sar_small': {'width': 48, 'depth': 2},
    'sar_base': {'width': 64, 'depth': 3},
}

def build_sar_ocr_model(*, in_channels: int, vocab_size: int, seq_len: int = 16, variant: str = 'sar_small', width_mult: float = 1.0) -> nn.Module:
    return build_toy_ocr_model(family='sar', variants=_VARIANTS, in_channels=int(in_channels), vocab_size=int(vocab_size), seq_len=int(seq_len), variant=str(variant), width_mult=float(width_mult), decoder_mode='gru')

if __name__ == '__main__':
    smoke_test_ocr(build_sar_ocr_model, 'sar_tiny')
