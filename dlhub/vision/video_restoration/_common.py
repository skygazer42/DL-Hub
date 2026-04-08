from __future__ import annotations
import torch
from torch import nn

def check_btchw(x):
    x=x.to(torch.float32)
    if x.ndim!=5: raise ValueError(f"Expected input shape (B,T,C,H,W), got {tuple(x.shape)}")
    return x
class ToyVideoRestorer(nn.Module):
    def __init__(self, *, family:str, in_channels:int, width:int, depth:int):
        super().__init__(); self.family=str(family); c=int(width); self.frame=nn.Sequential(nn.Conv2d(int(in_channels),c,3,1,1),nn.ReLU(inplace=True), nn.Conv2d(c,c,3,1,1),nn.ReLU(inplace=True)); self.head=nn.Conv2d(c,int(in_channels),3,1,1)
    def forward(self,video): x=check_btchw(video); b,t,c,h,w=x.shape; f=self.frame(x.view(b*t,c,h,w)); y=self.head(f).view(b,t,c,h,w); return {'restored': y}

def build_toy_video_restorer(*, family:str, variants:dict[str,dict[str,int]], in_channels:int, variant:str, width_mult:float=1.0):
    spec=variants[str(variant)]; width=max(16,int(int(spec['width'])*float(width_mult))); return ToyVideoRestorer(family=str(family), in_channels=int(in_channels), width=width, depth=int(spec['depth']))

def smoke_test_video(builder, variant:str):
    model=builder(in_channels=3, variant=variant, width_mult=0.5); out=model(torch.randn(2,4,3,64,64)); print(variant, tuple(out['restored'].shape))
