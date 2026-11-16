from runtime import *
import math
import torch
from torch import nn
from torch.nn import functional as F
from inputs import flatten

class ControlledAttention(nn.Module):
    def __init__(self,input_dim,output_dim,heads=2,direction='bidirectional',weighting='none'):
        super().__init__()
        if direction not in ('incoming','bidirectional','complete'):raise ValueError(direction)
        if weighting not in ('none','gate','normalized'):raise ValueError(weighting)
        self.direction,self.weighting,self.heads,self.output_dim=direction,weighting,heads,output_dim
        self.project=nn.Linear(input_dim,heads*output_dim,bias=False)
        self.target=nn.Parameter(torch.empty(heads,output_dim));self.source=nn.Parameter(torch.empty(heads,output_dim))
        # Same parameter shape across direction ablations. Reverse channels encode direction.
        self.edge=nn.Linear(4,heads,bias=False)
        nn.init.xavier_uniform_(self.target);nn.init.xavier_uniform_(self.source)
    def forward(self,x,edges,roles,return_weights=False):
        b,n,_=x.shape;v=self.project(x).reshape(b,n,self.heads,self.output_dim)
        reverse=edges.transpose(1,2)
        incoming=edges.abs().sum(-1)>0
        if self.direction=='incoming':mask=incoming
        else:mask=incoming | (reverse.abs().sum(-1)>0)
        mask=mask | torch.eye(n,dtype=torch.bool,device=x.device)[None]
        if self.direction=='complete':mask=torch.ones_like(mask)
        scores=F.leaky_relu((v*self.target).sum(-1)[:,:,None]+(v*self.source).sum(-1)[:,None,:]+self.edge(torch.cat([edges,reverse],-1)),.2)
        a=scores.masked_fill(~mask[...,None],-torch.inf).softmax(2)
        if self.weighting!='none':
            if not torch.all((roles>0)&(roles<=1)):raise ValueError('Resource weights outside (0,1]')
            a=a*roles[:,None,:,None]
            if self.weighting=='normalized':a=a/a.sum(2,keepdim=True)
        out=F.elu(torch.einsum('bijh,bjhd->bihd',a,v).mean(2))
        return (out,a) if return_weights else out

class ControlledModel(nn.Module):
    """G/N share temporal, draft, embedding, fusion, outcome loss; branch output=64."""
    def __init__(self,num_heroes,branch='G',direction='bidirectional',rank_input=True,weighting='none',missing_mask=True,width=64,dropout=.2):
        super().__init__()
        if branch not in ('G','N'):raise ValueError(branch)
        self.branch,self.rank_input,self.missing_mask=branch,rank_input,missing_mask
        self.embedding=nn.Embedding(num_heroes,16)
        self.ts_proj=nn.Linear(4,width)
        layer=nn.TransformerEncoderLayer(width,4,2*width,dropout,batch_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.ts_attention=nn.Sequential(nn.Linear(width,32),nn.Tanh(),nn.Linear(32,1))
        self.cross_r=nn.MultiheadAttention(16,4,batch_first=True);self.cross_d=nn.MultiheadAttention(16,4,batch_first=True)
        self.draft=nn.Sequential(nn.Linear(32,32),nn.ReLU())
        self.fusion=nn.Sequential(nn.Linear(width+32+64,64),nn.LayerNorm(64),nn.ReLU(),nn.Dropout(dropout))
        self.win=nn.Linear(64,1)
        if branch=='G':
            self.g1=ControlledAttention(43,width,direction=direction,weighting=weighting)
            self.g2=ControlledAttention(width,32,direction=direction,weighting=weighting)
        else:self.non_graph=nn.Sequential(nn.Linear(10*43+200,width),nn.ReLU(),nn.Dropout(dropout),nn.Linear(width,64))
    def forward(self,b):
        x=b['nodes'].clone();m=b['availability'] if self.missing_mask else torch.zeros_like(b['availability'])
        if not self.rank_input:x[...,13]=0
        h=self.embedding(b['heroes']);nodes=torch.cat([x,m,h],-1)
        r,d=h[:,:5],h[:,5:]
        ra,_=self.cross_r(r,d,d,need_weights=False);da,_=self.cross_d(d,r,r,need_weights=False)
        draft=self.draft(torch.cat([(r+ra).mean(1),(d+da).mean(1)],-1))
        z=self.ts_proj(b['sequence']);length=z.shape[1]
        positions=torch.arange(length,device=z.device,dtype=z.dtype)[:,None]
        div=torch.exp(torch.arange(0,z.shape[-1],2,device=z.device,dtype=z.dtype)*(-math.log(10000)/z.shape[-1]))
        pe=torch.zeros_like(z[0]);pe[:,0::2]=torch.sin(positions*div);pe[:,1::2]=torch.cos(positions*div)
        pad=torch.arange(length,device=z.device)[None]>=b['lengths'][:,None]
        z=self.temporal(z+pe,src_key_padding_mask=pad)
        a=self.ts_attention(z).squeeze(-1).masked_fill(pad,-torch.inf).softmax(1);ts=(z*a[...,None]).sum(1)
        if self.branch=='G':
            out=self.g2(self.g1(nodes,b['edges'],b['roles']),b['edges'],b['roles'])
            branch=torch.cat([out[:,:5].mean(1),out[:,5:].mean(1)],-1)
        else:branch=self.non_graph(torch.cat([nodes.flatten(1),b['edges'].flatten(1)],1))
        return self.win(self.fusion(torch.cat([ts,draft,branch],1))).squeeze(-1)

class FlatMLP(nn.Module):
    def __init__(self,num_heroes,minute,width=128,dropout=.2):
        super().__init__();self.num_heroes=num_heroes
        dim=200+70+200+4*(minute+1)+10*num_heroes
        self.net=nn.Sequential(nn.Linear(dim,width),nn.ReLU(),nn.Dropout(dropout),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,1))
    def forward(self,b):return self.net(flatten(b,self.num_heroes)).squeeze(-1)

def matrix():
    configs={}
    for direction in ('incoming','bidirectional','complete'):
        for rank in (False,True):
            for weight in ('none','gate','normalized'):
                name=f'G_{direction}_rank{int(rank)}_{weight}'
                configs[name]=dict(branch='G',direction=direction,rank_input=rank,weighting=weight)
    configs['G']=configs['G_bidirectional_rank1_none'].copy()
    configs['N']=dict(branch='N')
    configs['G_zero_missing_mask']=dict(branch='G',missing_mask=False)
    return configs
