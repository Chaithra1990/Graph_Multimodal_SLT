#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from graph_multimodal_slt.config import Config
from graph_multimodal_slt.data.datasets import make_splits, build_vocabs, SignLanguageDataset, slt_collate_fn
from graph_multimodal_slt.models.architecture import GraphMultimodalSLT

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['phoenix2014t','csldaily']); ap.add_argument('--config',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--sample-index',type=int,default=0); ap.add_argument('--output',required=True); ap.add_argument('--device',default='cuda'); args=ap.parse_args()
    cfg=Config.from_yaml(args.config); splits=make_splits(args.dataset,args.data_root); gv,tv=build_vocabs(splits['train']); ds=SignLanguageDataset(splits['test'],Path(args.data_root)/'features',cfg.max_seq_len,gv,tv,cfg.rgb_dim,cfg.pose_joints,cfg.face_dim); x=slt_collate_fn([ds[args.sample_index]]); device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu'); model=GraphMultimodalSLT(cfg,len(gv),len(tv),tv.pad_idx).to(device); ck=torch.load(args.checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck['model']); model.eval(); x={k:v.to(device) if torch.is_tensor(v) else v for k,v in x.items()};
    with torch.inference_mode(): _,_,alpha,tau=model(x['rgb'],x['pose'],x['face'],x['lengths'],x['text_input'])
    result={'sample_id':x['ids'][0],'rgb':alpha[0].cpu().tolist(),'temporal_saliency':tau[0,:,0].cpu().tolist()}; Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__': main()
