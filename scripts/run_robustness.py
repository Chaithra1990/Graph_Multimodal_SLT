#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from graph_multimodal_slt.config import Config
from graph_multimodal_slt.data.datasets import make_splits, build_vocabs, SignLanguageDataset, slt_collate_fn
from graph_multimodal_slt.models.architecture import GraphMultimodalSLT
from graph_multimodal_slt.training.engine import Trainer
from graph_multimodal_slt.utils.io import save_json

class Wrapped(torch.utils.data.Dataset):
    def __init__(self,base,condition): self.base=base; self.condition=condition
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        x=self.base[i];
        if self.condition=='mask_face': x['face']=torch.zeros_like(x['face'])
        elif self.condition=='jitter_face': x['face']=x['face']+torch.randn_like(x['face'])*0.01
        elif self.condition=='drop_frames':
            keep=torch.rand(x['length'])>0.3
            if not keep.any(): keep[0]=True
            x['rgb']=x['rgb'][keep]; x['pose']=x['pose'][keep]; x['face']=x['face'][keep]; x['length']=int(keep.sum())
        return x

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['phoenix2014t','csldaily']); ap.add_argument('--config',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
    cfg=Config.from_yaml(args.config); splits=make_splits(args.dataset,args.data_root); gv,tv=build_vocabs(splits['train']); base=SignLanguageDataset(splits['test'],Path(args.data_root)/'features',cfg.max_seq_len,gv,tv,cfg.rgb_dim,cfg.pose_joints,cfg.face_dim); device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); model=GraphMultimodalSLT(cfg,len(gv),len(tv),tv.pad_idx).to(device); ck=torch.load(args.checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck['model']); out={}
    for cond in ['clean','mask_face','jitter_face','drop_frames']:
        ds=base if cond=='clean' else Wrapped(base,cond); loader=DataLoader(ds,cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=slt_collate_fn); tr=Trainer(model,loader,loader,loader,gv,tv,cfg,Path(args.output_dir),device); m,_,_=tr.evaluate(loader); out[cond]=m
    save_json(out,Path(args.output_dir)/'robustness.json'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
