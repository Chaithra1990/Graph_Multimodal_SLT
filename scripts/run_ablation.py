#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from graph_multimodal_slt.config import Config
from graph_multimodal_slt.data.datasets import make_splits, build_vocabs, SignLanguageDataset, slt_collate_fn
from graph_multimodal_slt.models.variants import AblationModel
from graph_multimodal_slt.training.engine import Trainer
from graph_multimodal_slt.utils.seed import seed_everything
from graph_multimodal_slt.utils.io import save_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['phoenix2014t','csldaily']); ap.add_argument('--config',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--variant',required=True,choices=['without_face','without_pose','static_fusion','without_recognition_loss','without_temporal_saliency']); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--device',default='cuda'); args=ap.parse_args(); seed_everything(args.seed)
    cfg=Config.from_yaml(args.config); splits=make_splits(args.dataset,args.data_root); gv,tv=build_vocabs(splits['train']); root=Path(args.data_root)/'features'; device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); ds={k:SignLanguageDataset(splits[k],root,cfg.max_seq_len,gv,tv,cfg.rgb_dim,cfg.pose_joints,cfg.face_dim) for k in splits}; ld={k:DataLoader(ds[k],cfg.batch_size,shuffle=k=='train',num_workers=cfg.num_workers,collate_fn=slt_collate_fn) for k in splits}
    flags={'disable_face':args.variant=='without_face','disable_pose':args.variant=='without_pose','disable_saliency':args.variant=='without_temporal_saliency','static_fusion':args.variant=='static_fusion'}; model=AblationModel(cfg,len(gv),len(tv),tv.pad_idx,**flags).to(device); run=Path(args.output_dir)/args.variant; trainer=Trainer(model,ld['train'],ld['dev'],ld['test'],gv,tv,cfg,run,device)
    if args.variant=='without_recognition_loss': trainer.ctc=torch.nn.CTCLoss(blank=gv.pad_idx,zero_infinity=True); trainer._lambda_ctc=lambda epoch:0.0
    result=trainer.fit(); save_json({'variant':args.variant,'seed':args.seed,'metrics':result},run/'result.json'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
