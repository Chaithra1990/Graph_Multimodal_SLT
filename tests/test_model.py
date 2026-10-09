import torch
from graph_multimodal_slt.config import Config
from graph_multimodal_slt.models.architecture import GraphMultimodalSLT

def test_forward_shapes():
    c=Config(dataset='phoenix2014t',max_seq_len=16,d_model=64,d_ff=128,nhead=4,encoder_layers=1,decoder_layers=1,rgb_dim=32,face_dim=136,pose_joints=25)
    m=GraphMultimodalSLT(c,20,30,0)
    rgb=torch.randn(2,8,32); pose=torch.randn(2,8,25,2); face=torch.randn(2,8,136); lengths=torch.tensor([8,6]); txt=torch.randint(0,30,(2,5))
    g,t,a,tau=m(rgb,pose,face,lengths,txt)
    assert g.shape==(2,8,20); assert t.shape==(2,5,30); assert a.shape==(2,8,3); assert tau.shape==(2,8,1)
