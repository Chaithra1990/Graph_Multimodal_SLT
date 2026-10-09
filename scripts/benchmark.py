#!/usr/bin/env python3
import argparse, json, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from graph_multimodal_slt.config import Config
from graph_multimodal_slt.data.datasets import make_splits, build_vocabs, SignLanguageDataset, slt_collate_fn
from graph_multimodal_slt.models.architecture import GraphMultimodalSLT

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['phoenix2014t','csldaily']); ap.add_argument('--config',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--batches',type=int,default=20); args=ap.parse_args(); cfg=Config.from_yaml(args.config); splits=make_splits(args.dataset,args.data_root); gv,tv=build_vocabs(splits['train']); ds=SignLanguageDataset(splits['test'],Path(args.data_root)/'features',cfg.max_seq_len,gv,tv,cfg.rgb_dim,cfg.pose_joints,cfg.face_dim); ld=DataLoader(ds,cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=slt_collate_fn); device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); model=GraphMultimodalSLT(cfg,len(gv),len(tv),tv.pad_idx).to(device); ck=torch.load(args.checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck['model']); model.eval(); n=0; t0=time.perf_counter();
    with torch.inference_mode():
        for b in ld:
            if n>=args.batches: break
            b={k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}; _=model.greedy_decode(b['rgb'],b['pose'],b['face'],b['lengths'],tv.bos_idx,tv.eos_idx,cfg.max_decode_len); n+=b['rgb'].size(0)
    elapsed=time.perf_counter()-t0; fps=n/elapsed; params=sum(p.numel() for p in model.parameters()); print(json.dumps({'samples':n,'seconds':elapsed,'videos_per_second':fps,'frames_per_second_equivalent':fps*float(b['lengths'].float().mean()),'parameters':params},indent=2))
if __name__=='__main__': main()
