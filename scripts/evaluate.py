#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from graph_multimodal_slt.config import Config
from graph_multimodal_slt.data.datasets import make_splits, build_vocabs, SignLanguageDataset, slt_collate_fn
from graph_multimodal_slt.data.vocab import Vocab
from graph_multimodal_slt.models.architecture import GraphMultimodalSLT
from graph_multimodal_slt.training.engine import Trainer
from graph_multimodal_slt.utils.io import save_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['phoenix2014t','csldaily']); ap.add_argument('--config',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--split',default='test',choices=['dev','test']); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
    cfg=Config.from_yaml(args.config); splits=make_splits(args.dataset,args.data_root); gv,tv=build_vocabs(splits['train']); ds=SignLanguageDataset(splits[args.split],Path(args.data_root)/'features',cfg.max_seq_len,gv,tv,cfg.rgb_dim,cfg.pose_joints,cfg.face_dim); device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); loader=DataLoader(ds,cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=slt_collate_fn); model=GraphMultimodalSLT(cfg,len(gv),len(tv),tv.pad_idx).to(device); ck=torch.load(args.checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck['model']); trainer=Trainer(model,loader,loader,loader,gv,tv,cfg,Path(args.checkpoint).parent,device); metrics,refs,hyps=trainer.evaluate(loader); out={'split':args.split,'metrics':metrics,'references':refs,'hypotheses':hyps}; save_json(out,Path(args.checkpoint).parent/f'{args.split}_evaluation.json'); print(json.dumps({'split':args.split,'metrics':metrics},indent=2))
if __name__=='__main__': main()
