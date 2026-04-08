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
    def __init__(self, *, family:str, in_channels:int, width:int, depth:int, num_classes:int=10):
        super().__init__(); self.family=str(family); self.enc=TinyEncoder(in_channels,width,depth); c=self.enc.out_channels; self.det=nn.Conv2d(c,1,1); self.embed=nn.Linear(c,128); self.cls=nn.Linear(128,int(num_classes))
    def forward(self,image): feat=self.enc(image); pooled=F.adaptive_avg_pool2d(feat,(1,1)).flatten(1); emb=F.normalize(self.embed(pooled),dim=1); return {'score_map': self.det(feat), 'embedding': emb, 'logits': self.cls(emb)}

def build_toy_model(*, family:str, variants:dict[str,dict[str,int]], in_channels:int, variant:str, width_mult:float=1.0, num_classes:int=10, **kwargs):
    spec=variants[str(variant)]; width=max(16,int(int(spec['width'])*float(width_mult))); return ToyModel(family=str(family), in_channels=int(in_channels), width=width, depth=int(spec['depth']), num_classes=int(num_classes))

def smoke_test_model(builder, variant:str): out=builder(in_channels=3,variant=variant,width_mult=0.5,num_classes=6)(torch.randn(2,3,128,64)); print(variant,{k:tuple(v.shape) for k,v in out.items()})
