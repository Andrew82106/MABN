import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

class Probe(nn.Module):
    def __init__(self,dim,architecture='mlp',width=64,depth=1,dropout=.1):
        super().__init__();self.architecture=architecture
        if architecture=='linear':self.body=nn.Linear(dim,1)
        elif architecture=='mlp':
            layers=[]
            for _ in range(depth):layers.extend([nn.Linear(dim,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(dropout)]);dim=width
            self.body=nn.Sequential(*layers,nn.Linear(width,1))
        else:
            self.proj=nn.Sequential(nn.Linear(dim,width),nn.LayerNorm(width),nn.GELU())
            if architecture=='conv':self.sequence=nn.Conv1d(width,width,kernel_size=5,padding=4)
            else:self.sequence=nn.GRU(width,width,batch_first=True)
            self.drop=nn.Dropout(dropout);self.head=nn.Linear(width,1)
    def forward(self,x):
        if self.architecture in ['linear','mlp']:return self.body(x).squeeze(-1)
        z=self.proj(x)
        if self.architecture=='conv':z=torch.nn.functional.gelu(self.sequence(z.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        else:z,_=self.sequence(z)
        return self.head(self.drop(z)).squeeze(-1)

def batch(xx,device):
    lengths=[len(x) for x in xx]
    return pad_sequence([torch.as_tensor(x,device=device) for x in xx],batch_first=True),lengths

@torch.no_grad()
def predict(net,xx):
    net.eval();out=[];device=next(net.parameters()).device
    for i in range(0,len(xx),16):
        x,lengths=batch(xx[i:i+16],device);ss=net(x).sigmoid().cpu().numpy()
        out.extend([s[:n] for s,n in zip(ss,lengths)])
    return out
