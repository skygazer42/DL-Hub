from __future__ import annotations
import torch
from torch import nn

def check_nchw(x):
    x=x.to(torch.float32)
    if x.ndim!=4: raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x
class ToyInteractiveSeg(nn.Module):
    def __init__(self, *, family:str, in_channels:int, width:int, depth:int):
        super().__init__(); self.family=str(family); c=int(width); self.backbone=nn.Sequential(nn.Conv2d(int(in_channels)+1,c,3,1,1),nn.ReLU(inplace=True), *sum([[nn.Conv2d(c,c,3,1,1),nn.ReLU(inplace=True)] for _ in range(max(1,int(depth)))], [])); self.head=nn.Conv2d(c,1,1)
    def forward(self,image,prompt=None): x=check_nchw(image); p=torch.zeros(x.shape[0],1,x.shape[2],x.shape[3],device=x.device,dtype=x.dtype) if prompt is None else prompt.to(x.dtype); logits=self.head(self.backbone(torch.cat([x,p],dim=1))); return {'logits': logits, 'mask': torch.sigmoid(logits)}

def build_toy_inter(*, family:str, variants:dict[str,dict[str,int]], in_channels:int, variant:str, width_mult:float=1.0):
    spec=variants[str(variant)]; width=max(16,int(int(spec['width'])*float(width_mult))); return ToyInteractiveSeg(family=str(family), in_channels=int(in_channels), width=width, depth=int(spec['depth']))

def smoke_test_inter(builder, variant:str): out=builder(in_channels=3, variant=variant, width_mult=0.5)(torch.randn(2,3,64,64)); print(variant, tuple(out['mask'].shape))
