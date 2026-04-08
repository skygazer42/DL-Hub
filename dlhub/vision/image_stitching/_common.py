from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def check_nchw(x):
    x=x.to(torch.float32)
    if x.ndim!=4: raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x
class TinyEncoder(nn.Module):
    def __init__(self,in_channels:int,width:int,depth:int):
        super().__init__(); c=int(width); layers=[nn.Conv2d(int(in_channels),c,3,1,1),nn.ReLU(inplace=True)]
        for _ in range(max(1,int(depth))): layers += [nn.Conv2d(c,c,3,1,1),nn.ReLU(inplace=True)]
        self.net=nn.Sequential(*layers); self.out_channels=c
    def forward(self,x): return self.net(check_nchw(x))

class ToyModel(nn.Module):
    def __init__(self, *, family:str, in_channels:int, width:int, depth:int):
        super().__init__(); self.family=str(family); self.enc=TinyEncoder(in_channels*2,width,depth); self.homo=nn.Linear(self.enc.out_channels,8)
    def forward(self, image1, image2): x=torch.cat([check_nchw(image1), check_nchw(image2)], dim=1); feat=self.enc(x); pooled=F.adaptive_avg_pool2d(feat,(1,1)).flatten(1); offsets=self.homo(pooled).view(image1.shape[0],4,2); return {'offsets': offsets}

def build_toy_model(*, family:str, variants:dict[str,dict[str,int]], in_channels:int, variant:str, width_mult:float=1.0, **kwargs):
    spec=variants[str(variant)]; width=max(16,int(int(spec['width'])*float(width_mult))); return ToyModel(family=str(family), in_channels=int(in_channels), width=width, depth=int(spec['depth']))

def smoke_test_model(builder, variant:str):
    x1=torch.randn(2,3,128,128); x2=torch.randn(2,3,128,128); out=builder(in_channels=3, variant=variant, width_mult=0.5)(x1,x2); print(variant, tuple(out['offsets'].shape))
