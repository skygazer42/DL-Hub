from __future__ import annotations
import torch
from torch import nn

def check_nchw(x):
    x=x.to(torch.float32)
    if x.ndim!=4: raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x
class ToyChangeDetector(nn.Module):
    def __init__(self, *, family:str, in_channels:int, width:int, depth:int):
        super().__init__(); self.family=str(family); c=int(width); self.backbone=nn.Sequential(nn.Conv2d(int(in_channels)*2,c,3,1,1),nn.ReLU(inplace=True), *sum([[nn.Conv2d(c,c,3,1,1),nn.ReLU(inplace=True)] for _ in range(max(1,int(depth)))], [])); self.head=nn.Conv2d(c,1,1)
    def forward(self,image1,image2): x=torch.cat([check_nchw(image1),check_nchw(image2)],dim=1); logits=self.head(self.backbone(x)); mask=torch.sigmoid(logits); return {'logits': logits, 'mask': mask}

def build_toy_change(*, family:str, variants:dict[str,dict[str,int]], in_channels:int, variant:str, width_mult:float=1.0):
    spec=variants[str(variant)]; width=max(16,int(int(spec['width'])*float(width_mult))); return ToyChangeDetector(family=str(family), in_channels=int(in_channels), width=width, depth=int(spec['depth']))

def smoke_test_change(builder, variant:str): out=builder(in_channels=3, variant=variant, width_mult=0.5)(torch.randn(2,3,128,128), torch.randn(2,3,128,128)); print(variant,{k:tuple(v.shape) for k,v in out.items()})
