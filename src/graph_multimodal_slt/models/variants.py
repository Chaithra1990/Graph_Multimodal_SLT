import torch
import torch.nn as nn
from .architecture import GraphMultimodalSLT

class StaticConcatFusion(nn.Module):
    def __init__(self,d): super().__init__(); self.proj=nn.Linear(3*d,d)
    def forward(self,a,b,c):
        x=self.proj(torch.cat([a,b,c],-1)); alpha=torch.full((*x.shape[:2],3),1/3,device=x.device); return x,alpha

class AblationModel(GraphMultimodalSLT):
    def __init__(self,*args,disable_face=False,disable_pose=False,disable_saliency=False,static_fusion=False,**kwargs):
        super().__init__(*args,**kwargs); self.disable_face=disable_face; self.disable_pose=disable_pose; self.disable_saliency=disable_saliency
        if static_fusion: self.fusion=StaticConcatFusion(args[0].d_model)
    def encode(self,rgb,pose,face,lengths):
        mask=torch.arange(rgb.size(1),device=rgb.device)[None,:]<lengths[:,None]
        e1,e2,e3=self.rgb(rgb),self.pose(pose),self.face(face)
        if self.disable_pose: e2=torch.zeros_like(e2)
        if self.disable_face: e3=torch.zeros_like(e3)
        fused,alpha=self.fusion(e1,e2,e3)
        if self.disable_saliency: salient=fused; tau=mask.float().unsqueeze(-1)
        else: salient,tau=self.saliency(fused,mask)
        enc=self.encoder(self.pos(salient),src_key_padding_mask=~mask); return enc,mask,alpha,tau
