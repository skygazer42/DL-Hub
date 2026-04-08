from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def check_nchw(x):
    x=x.to(torch.float32)
    if x.ndim!=4: raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x
class TinyEncoder(nn.Module):
    def __init__(self, in_channels:int, width:int, depth:int):
        super().__init__(); c=int(width); layers=[nn.Conv2d(int(in_channels),c,3,2,1),nn.ReLU(inplace=True)]
        for _ in range(max(1,int(depth))): layers += [nn.Conv2d(c,c,3,1,1),nn.ReLU(inplace=True)]
        self.net=nn.Sequential(*layers); self.out_channels=c
    def forward(self,x): return self.net(check_nchw(x))

class ToyAttributeRecognizer(nn.Module):
    def __init__(self, *, family:str, in_channels:int, num_attributes:int, width:int, depth:int):
        super().__init__(); self.family=str(family); self.enc=TinyEncoder(in_channels,width,depth); self.cls=nn.Linear(self.enc.out_channels,int(num_attributes))
    def forward(self,image): feat=self.enc(image); pooled=F.adaptive_avg_pool2d(feat,(1,1)).flatten(1); logits=self.cls(pooled); return {'attribute_logits': logits}

def build_toy_attr(*, family:str, variants:dict[str,dict[str,int]], in_channels:int, num_attributes:int, variant:str, width_mult:float=1.0):
    spec=variants[str(variant)]; width=max(16,int(int(spec['width'])*float(width_mult))); return ToyAttributeRecognizer(family=str(family), in_channels=int(in_channels), num_attributes=int(num_attributes), width=width, depth=int(spec['depth']))

def smoke_test_attr(builder, variant:str):
    model=builder(in_channels=3, num_attributes=12, variant=variant, width_mult=0.5); out=model(torch.randn(2,3,128,64)); print(variant, tuple(out['attribute_logits'].shape))
