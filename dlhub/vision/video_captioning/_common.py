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
    def __init__(self, *, family:str, in_channels:int, width:int, depth:int, vocab_size:int=64, seq_len:int=12):
        super().__init__(); self.family=str(family); self.seq_len=int(seq_len); self.frame=TinyEncoder(in_channels,width,depth); c=self.frame.out_channels; self.rnn=nn.GRU(c,c,batch_first=True); self.head=nn.Linear(c,int(vocab_size))
    def forward(self,video): video=video.to(torch.float32); b,t,c,h,w=video.shape; feat=self.frame(video.view(b*t,c,h,w)).mean(dim=(2,3)).view(b,t,-1); seq,_=self.rnn(feat); logits=self.head(seq[:, : self.seq_len]); return {'logits': logits, 'tokens': logits.argmax(dim=-1)}

def build_toy_model(*, family:str, variants:dict[str,dict[str,int]], in_channels:int, variant:str, width_mult:float=1.0, vocab_size:int=64, seq_len:int=12, **kwargs):
    spec=variants[str(variant)]; width=max(16,int(int(spec['width'])*float(width_mult))); return ToyModel(family=str(family), in_channels=int(in_channels), width=width, depth=int(spec['depth']), vocab_size=int(vocab_size), seq_len=int(seq_len))

def smoke_test_model(builder, variant:str): out=builder(in_channels=3,variant=variant,width_mult=0.5,vocab_size=64,seq_len=12)(torch.randn(2,4,3,64,64)); print(variant,{k:tuple(v.shape) for k,v in out.items()})
