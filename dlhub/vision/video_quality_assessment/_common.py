from __future__ import annotations
import torch
from torch import nn

def check_btchw(x):
    x = x.to(torch.float32)
    if x.ndim != 5:
        raise ValueError(f"Expected input shape (B,T,C,H,W), got {tuple(x.shape)}")
    return x

class ToyModel(nn.Module):
    def __init__(self, *, family:str, in_channels:int, width:int, depth:int):
        super().__init__()
        self.family = str(family)
        c = int(width)
        self.frame = nn.Sequential(nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True))
        self.rnn = nn.GRU(c, c, batch_first=True)
        self.head = nn.Linear(c, 1)

    def forward(self, video):
        x = check_btchw(video)
        b,t,c,h,w = x.shape
        feat = self.frame(x.view(b*t,c,h,w)).mean(dim=(2,3)).view(b,t,-1)
        seq,_ = self.rnn(feat)
        score = self.head(seq[:, -1]).squeeze(-1)
        return {'score': score}

def build_toy_model(*, family:str, variants:dict[str,dict[str,int]], in_channels:int, variant:str, width_mult:float=1.0, **kwargs):
    spec = variants[str(variant)]
    width = max(16, int(int(spec['width']) * float(width_mult)))
    return ToyModel(family=str(family), in_channels=int(in_channels), width=width, depth=int(spec['depth']))

def smoke_test_model(builder, variant:str):
    out = builder(in_channels=3, variant=variant, width_mult=0.5)(torch.randn(2,4,3,64,64))
    print(variant, tuple(out['score'].shape))
