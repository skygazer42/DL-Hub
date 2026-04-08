from __future__ import annotations
from ._common import build_toy_expr, smoke_test_expr
_VARIANTS = {'emotionvit_tiny': {'width':24,'depth':1}, 'emotionvit_small': {'width':32,'depth':2}, 'emotionvit_base': {'width':48,'depth':3}}
def build_emotionvit_expression_recognizer(*, in_channels:int, num_classes:int, variant:str='emotionvit_small', width_mult:float=1.0):
    return build_toy_expr(family='emotionvit', variants=_VARIANTS, in_channels=int(in_channels), num_classes=int(num_classes), variant=str(variant), width_mult=float(width_mult))
if __name__ == '__main__': smoke_test_expr(build_emotionvit_expression_recognizer, 'emotionvit_tiny')
