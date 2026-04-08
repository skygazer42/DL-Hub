from __future__ import annotations
from ._common import build_toy_video_qa, smoke_test_video_qa
_VARIANTS = {'hme_qa_tiny': {'width':24,'depth':1}, 'hme_qa_small': {'width':32,'depth':2}, 'hme_qa_base': {'width':48,'depth':3}}

def build_hme_qa_video_qa_model(*, in_channels:int, variant:str='hme_qa_small', width_mult:float=1.0, answer_vocab:int=32):
    return build_toy_video_qa(family='hme_qa', variants=_VARIANTS, in_channels=int(in_channels), variant=str(variant), width_mult=float(width_mult), answer_vocab=int(answer_vocab))

if __name__ == '__main__': smoke_test_video_qa(build_hme_qa_video_qa_model, 'hme_qa_tiny')
