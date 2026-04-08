from __future__ import annotations
import torch
from torch import nn

def check_nchw(x):
    x=x.to(torch.float32)
    if x.ndim!=4: raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x
class ToyFaceDetector(nn.Module):
    def __init__(self, *, family:str, in_channels:int, width:int, depth:int):
        super().__init__(); self.family=str(family); c=int(width); layers=[nn.Conv2d(int(in_channels),c,3,2,1),nn.ReLU(inplace=True)]
        for _ in range(max(1,int(depth))): layers += [nn.Conv2d(c,c,3,1,1),nn.ReLU(inplace=True)]
        self.net=nn.Sequential(*layers); self.cls=nn.Conv2d(c,1,1); self.box=nn.Conv2d(c,4,1); self.landmark=nn.Conv2d(c,10,1)
    def forward(self,image): feat=self.net(check_nchw(image)); return {'score_map': self.cls(feat), 'boxes': self.box(feat), 'landmarks': self.landmark(feat)}

def build_toy_face_detector(*, family:str, variants:dict[str,dict[str,int]], in_channels:int, variant:str, width_mult:float=1.0):
    spec=variants[str(variant)]; width=max(16,int(int(spec['width'])*float(width_mult))); return ToyFaceDetector(family=str(family), in_channels=int(in_channels), width=width, depth=int(spec['depth']))

def smoke_test_fd(builder, variant:str):
    model=builder(in_channels=3, variant=variant, width_mult=0.5); out=model(torch.randn(2,3,128,128)); print(variant, {k:tuple(v.shape) for k,v in out.items()})
