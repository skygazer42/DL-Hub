from __future__ import annotations
import torch
from torch import nn

def check_bnc(x):
    x=x.to(torch.float32)
    if x.ndim!=3: raise ValueError(f"Expected input shape (B,N,C), got {tuple(x.shape)}")
    return x
class ToyRegistrar(nn.Module):
    def __init__(self, *, family:str, width:int, depth:int):
        super().__init__(); self.family=str(family); c=int(width); self.proj=nn.Linear(3,c); self.rnn=nn.GRU(c,c,batch_first=True); self.head=nn.Linear(c,6)
    def forward(self, src, tgt): s=self.proj(check_bnc(src)); t=self.proj(check_bnc(tgt)); _,hs=self.rnn(torch.cat([s,t],dim=1)); pose=self.head(hs[-1]); return {'pose6d': pose}

def build_toy_model(*, family:str, variants:dict[str,dict[str,int]], variant:str, width_mult:float=1.0, **kwargs):
    spec=variants[str(variant)]; width=max(16,int(int(spec['width'])*float(width_mult))); return ToyRegistrar(family=str(family), width=width, depth=int(spec['depth']))

def smoke_test_model(builder, variant:str): out=builder(variant=variant, width_mult=0.5)(torch.randn(2,128,3), torch.randn(2,128,3)); print(variant, tuple(out['pose6d'].shape))
