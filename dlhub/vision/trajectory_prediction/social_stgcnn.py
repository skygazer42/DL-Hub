from __future__ import annotations
from ._common import build_toy_model, smoke_test_model
_VARIANTS = {'social_stgcnn_tiny': {'width':24,'depth':1}, 'social_stgcnn_small': {'width':32,'depth':2}, 'social_stgcnn_base': {'width':48,'depth':3}}
def build_social_stgcnn_trajectory_predictor(*, coord_dim:int, variant:str='social_stgcnn_small', width_mult:float=1.0, pred_steps:int=12):
    return build_toy_model(family='social_stgcnn', variants=_VARIANTS, coord_dim=int(coord_dim), variant=str(variant), width_mult=float(width_mult), pred_steps=int(pred_steps))
if __name__ == '__main__': smoke_test_model(build_social_stgcnn_trajectory_predictor, 'social_stgcnn_tiny')
