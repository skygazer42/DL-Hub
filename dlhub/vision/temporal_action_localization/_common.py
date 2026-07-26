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
    def __init__(self, *, family:str, in_channels:int, width:int, depth:int, num_classes:int=5):
        super().__init__(); self.family=str(family); self.proj=nn.Linear(int(in_channels), int(width)); self.temporal=nn.GRU(int(width), int(width), batch_first=True); self.cls=nn.Linear(int(width), int(num_classes)); self.boundary=nn.Linear(int(width),2)
    def forward(self, video_feat):
        x=video_feat.to(torch.float32)
        if x.ndim!=3: raise ValueError(f"Expected input shape (B,T,C), got {tuple(x.shape)}")
        tok=self.proj(x); seq,_=self.temporal(tok)
        return {'class_logits': self.cls(seq), 'boundaries': torch.sigmoid(self.boundary(seq))}

def build_toy_model(*, family:str, variants:dict[str,dict[str,int]], in_channels:int, variant:str, width_mult:float=1.0, num_classes:int=5, **kwargs):
    spec=variants[str(variant)]; width=max(16,int(int(spec['width'])*float(width_mult))); return ToyModel(family=str(family), in_channels=int(in_channels), width=width, depth=int(spec['depth']), num_classes=int(num_classes))

def smoke_test_model(builder, variant:str):
    out=builder(in_channels=64, variant=variant, width_mult=0.5)(torch.randn(2,16,64)); print(variant, {k:tuple(v.shape) for k,v in out.items()})
