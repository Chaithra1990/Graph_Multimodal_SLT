#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from graph_multimodal_slt.config import Config
from graph_multimodal_slt.data.datasets import make_splits, build_vocabs, SignLanguageDataset, slt_collate_fn
from graph_multimodal_slt.models.architecture import GraphMultimodalSLT
from graph_multimodal_slt.training.engine import Trainer
from graph_multimodal_slt.utils.seed import seed_everything
from graph_multimodal_slt.utils.io import save_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['phoenix2014t','csldaily']); ap.add_argument('--config',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--seeds',nargs='+',type=int); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
    cfg=Config.from_yaml(args.config); seeds=args.seeds or list(cfg.seeds); root=Path(args.data_root); feature_root=root/'features'; splits=make_splits(args.dataset,root); gv,tv=build_vocabs(splits['train'])
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    summary=[]
    for seed in seeds:
        seed_everything(seed); ds={k:SignLanguageDataset(splits[k],feature_root,cfg.max_seq_len,gv,tv,cfg.rgb_dim,cfg.pose_joints,cfg.face_dim) for k in splits}
        loaders={k:DataLoader(ds[k],cfg.batch_size,shuffle=k=='train',num_workers=cfg.num_workers,pin_memory=device.type=='cuda',collate_fn=slt_collate_fn) for k in splits}
        model=GraphMultimodalSLT(cfg,len(gv),len(tv),tv.pad_idx).to(device)
        run=Path(args.output_dir)/f'seed_{seed}'; save_json({'config':cfg.to_dict(),'seed':seed,'dataset':args.dataset,'vocabularies':{'gloss':gv.to_dict(),'text':tv.to_dict()}},run/'run_config.json')
        trainer=Trainer(model,loaders['train'],loaders['dev'],loaders['test'],gv,tv,cfg,run,device); result=trainer.fit(); summary.append({'seed':seed,**result}); print(json.dumps({'seed':seed,**result},indent=2))
    save_json(summary,Path(args.output_dir)/'multi_seed_results.json')
if __name__=='__main__': main()
