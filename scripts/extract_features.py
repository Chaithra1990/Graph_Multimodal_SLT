#!/usr/bin/env python3
import argparse
from pathlib import Path
from graph_multimodal_slt.config import Config
from graph_multimodal_slt.data.datasets import make_splits
from graph_multimodal_slt.data.extraction import RGBExtractor, extract_sample

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['phoenix2014t','csldaily']); ap.add_argument('--config',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--openpose-json-root',required=True); ap.add_argument('--output-root',required=True); ap.add_argument('--device',default='cuda'); ap.add_argument('--only-split',choices=['train','dev','test']); args=ap.parse_args()
    cfg=Config.from_yaml(args.config); splits=make_splits(args.dataset,args.data_root)
    ex=RGBExtractor(args.device,cfg.frame_size,cfg.rgb_weights)
    split_names=[args.only_split] if args.only_split else ['train','dev','test']
    for split in split_names:
        for s in splits[split]:
            if not s.video: raise RuntimeError('Video path resolution failed; provide videos/ under data-root')
            op=Path(args.openpose_json_root)/s.sample_id
            out=Path(args.output_root)/f'{s.sample_id}.pt'
            if out.exists(): continue
            extract_sample(s.video,op,out,ex,s.sample_id,cfg.frame_stride)
            print(f'Wrote {out}')
if __name__=='__main__': main()
