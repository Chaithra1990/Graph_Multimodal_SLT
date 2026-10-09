from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGraphConv(nn.Module):
    def __init__(self,in_channels,out_channels,num_joints=25,adjacency=None):
        super().__init__(); self.num_joints=num_joints
        if adjacency is None:
            A=torch.eye(num_joints)
            # OpenPose BODY_25 physical edges.
            edges=[(1,8),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,0),(0,15),(15,17),(0,16),(16,18),(8,9),(9,10),(10,11),(8,12),(12,13),(13,14),(11,24),(11,22),(22,23),(14,21),(14,19),(19,20)]
            for a,b in edges:
                if a<num_joints and b<num_joints: A[a,b]=A[b,a]=1
        else: A=torch.as_tensor(adjacency,dtype=torch.float32)
        self.register_buffer('A_tilde',A)
        self.G=nn.Parameter(torch.ones(num_joints,num_joints)); self.weight=nn.Parameter(torch.empty(in_channels,out_channels)); nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
    def forward(self,x):
        B,T,V,C=x.shape; A=self.A_tilde*self.G
        d=A.sum(1).clamp_min(1e-6); inv=d.rsqrt(); A=inv[:,None]*A*inv[None,:]
        y=torch.matmul(x.reshape(B*T,V,C),self.weight); y=torch.matmul(A,y); return F.relu(y).reshape(B,T,V,-1)

class PoseGCNBranch(nn.Module):
    def __init__(self,in_channels=2,hidden=64,out_dim=512,num_joints=25):
        super().__init__(); self.g1=SpatialGraphConv(in_channels,hidden,num_joints); self.g2=SpatialGraphConv(hidden,hidden*2,num_joints); self.g3=SpatialGraphConv(hidden*2,hidden*4,num_joints); self.proj=nn.Linear(num_joints*hidden*4,out_dim)
    def forward(self,x):
        x=self.g1(x); x=self.g2(x); x=self.g3(x); return self.proj(x.reshape(x.size(0),x.size(1),-1))

class FeatureProjectionBranch(nn.Module):
    def __init__(self,in_dim,out_dim): super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,out_dim),nn.LayerNorm(out_dim),nn.ReLU(),nn.Dropout(.1))
    def forward(self,x): return self.net(x)

class ConfidenceAwareFusion(nn.Module):
    def __init__(self,d=512,gamma=.5):
        super().__init__(); self.gamma=gamma
        self.W=nn.ModuleList([nn.Linear(d,d) for _ in range(3)]); self.w=nn.ModuleList([nn.Linear(d,1,bias=False) for _ in range(3)])
    def forward(self,*xs):
        scores=[self.w[i](torch.tanh(self.W[i](xs[i]))) for i in range(3)]
        z=[F.normalize(x,p=2,dim=-1) for x in xs]
        sim=[[None]*3 for _ in range(3)]
        for i in range(3):
            for j in range(i+1,3): sim[i][j]=sim[j][i]=(z[i]*z[j]).sum(-1,keepdim=True)
        consistency=[sim[0][1]+sim[0][2],sim[0][1]+sim[1][2],sim[0][2]+sim[1][2]]
        alpha=F.softmax(torch.cat([scores[i]+self.gamma*consistency[i] for i in range(3)],-1),-1)
        fused=sum(alpha[...,i:i+1]*xs[i] for i in range(3)); return fused,alpha

class TemporalSaliencyGate(nn.Module):
    def __init__(self,d=512): super().__init__(); self.net=nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,1,bias=False))
    def forward(self,x,mask):
        s=self.net(x).squeeze(-1).masked_fill(~mask,-1e9); tau=F.softmax(s,-1).unsqueeze(-1); return x*tau,tau

class PositionalEncoding(nn.Module):
    def __init__(self,d,max_len=300):
        super().__init__(); pe=torch.zeros(max_len,d); pos=torch.arange(max_len).float()[:,None]; div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.)/d)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div); self.register_buffer('pe',pe[None])
    def forward(self,x): return x+self.pe[:,:x.size(1)]

class GraphMultimodalSLT(nn.Module):
    def __init__(self,cfg,gloss_vocab_size,text_vocab_size,pad_idx):
        super().__init__(); d=cfg.d_model
        self.pose=PoseGCNBranch(cfg.pose_channels,64,d,cfg.pose_joints); self.rgb=FeatureProjectionBranch(cfg.rgb_dim,d); self.face=FeatureProjectionBranch(cfg.face_dim,d)
        self.fusion=ConfidenceAwareFusion(d,cfg.fusion_gamma); self.saliency=TemporalSaliencyGate(d); self.pos=PositionalEncoding(d,cfg.max_seq_len)
        enc=nn.TransformerEncoderLayer(d,cfg.nhead,cfg.d_ff,cfg.dropout,batch_first=True); self.encoder=nn.TransformerEncoder(enc,cfg.encoder_layers)
        self.gloss=nn.Linear(d,gloss_vocab_size)
        self.emb=nn.Embedding(text_vocab_size,d,padding_idx=pad_idx); dec=nn.TransformerDecoderLayer(d,cfg.nhead,cfg.d_ff,cfg.dropout,batch_first=True); self.decoder=nn.TransformerDecoder(dec,cfg.decoder_layers); self.text=nn.Linear(d,text_vocab_size)
    def encode(self,rgb,pose,face,lengths):
        mask=torch.arange(rgb.size(1),device=rgb.device)[None,:]<lengths[:,None]
        e1,e2,e3=self.rgb(rgb),self.pose(pose),self.face(face); fused,alpha=self.fusion(e1,e2,e3); fused,tau=self.saliency(fused,mask); enc=self.encoder(self.pos(fused),src_key_padding_mask=~mask); return enc,mask,alpha,tau
    def forward(self,rgb,pose,face,lengths,text_input):
        mem,mask,a,t=self.encode(rgb,pose,face,lengths); gloss=self.gloss(mem); y=self.pos(self.emb(text_input)); n=y.size(1); causal=torch.triu(torch.full((n,n),float('-inf'),device=y.device),1); dec=self.decoder(y,mem,tgt_mask=causal,memory_key_padding_mask=~mask); return gloss,self.text(dec),a,t
    @torch.no_grad()
    def greedy_decode(self,rgb,pose,face,lengths,bos_idx,eos_idx,max_len=80):
        mem,mask,_,_=self.encode(rgb,pose,face,lengths); ys=torch.full((rgb.size(0),1),bos_idx,dtype=torch.long,device=rgb.device); done=torch.zeros(rgb.size(0),dtype=torch.bool,device=rgb.device)
        for _ in range(max_len-1):
            y=self.pos(self.emb(ys)); n=y.size(1); causal=torch.triu(torch.full((n,n),float('-inf'),device=y.device),1); dec=self.decoder(y,mem,tgt_mask=causal,memory_key_padding_mask=~mask); nxt=self.text(dec[:,-1]).argmax(-1,keepdim=True); ys=torch.cat([ys,nxt],1); done|=nxt[:,0].eq(eos_idx)
            if done.all(): break
        return ys

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self,smoothing=.1,pad_idx=0): super().__init__(); self.smoothing=smoothing; self.pad_idx=pad_idx
    def forward(self,logits,target):
        lp=F.log_softmax(logits,-1); nll=-lp.gather(-1,target.unsqueeze(-1)).squeeze(-1); smooth=-lp.mean(-1); loss=(1-self.smoothing)*nll+self.smoothing*smooth; m=target.ne(self.pad_idx); return loss[m].mean()
