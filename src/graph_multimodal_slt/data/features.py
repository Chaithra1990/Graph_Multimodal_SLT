from __future__ import annotations
from pathlib import Path
import torch

REQUIRED=("rgb","pose","face")

def feature_path(root, sample_id): return Path(root) / f"{sample_id}.pt"

def load_feature_file(root, sample_id, max_seq_len, rgb_dim=2048, pose_joints=25, face_dim=136):
    p=feature_path(root,sample_id)
    if not p.is_file():
        raise FileNotFoundError(f"Missing real extracted feature file for {sample_id}: {p}. Run scripts/extract_features.py first.")
    d=torch.load(p,map_location="cpu",weights_only=True)
    missing=[k for k in REQUIRED if k not in d]
    if missing: raise ValueError(f"Feature file {p} is missing {missing}")
    rgb,pose,face=d["rgb"],d["pose"],d["face"]
    if rgb.ndim!=2 or rgb.shape[1]!=rgb_dim: raise ValueError(f"{p}: rgb must be [T,{rgb_dim}], got {tuple(rgb.shape)}")
    if pose.ndim!=3 or pose.shape[1:]!=(pose_joints,2): raise ValueError(f"{p}: pose must be [T,{pose_joints},2], got {tuple(pose.shape)}")
    if face.ndim!=2 or face.shape[1]!=face_dim: raise ValueError(f"{p}: face must be [T,{face_dim}], got {tuple(face.shape)}")
    lengths=min(rgb.shape[0],pose.shape[0],face.shape[0],max_seq_len)
    if lengths<=0: raise ValueError(f"{p}: empty sequence")
    return {"rgb":rgb[:lengths].float(),"pose":pose[:lengths].float(),"face":face[:lengths].float(),"length":lengths}
