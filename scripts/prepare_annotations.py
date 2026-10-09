#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
from graph_multimodal_slt.config import Config
from graph_multimodal_slt.data.datasets import make_splits, build_vocabs
from graph_multimodal_slt.data.annotations import validate_official_counts
from graph_multimodal_slt.utils.io import save_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['phoenix2014t','csldaily']); ap.add_argument('--config',required=True); ap.add_argument('--data-root',required=True); args=ap.parse_args()
    cfg=Config.from_yaml(args.config); splits=make_splits(args.dataset,args.data_root); validate_official_counts(args.dataset,splits)
    gv,tv=build_vocabs(splits['train'])
    out=Path(args.data_root)/'prepared'; out.mkdir(exist_ok=True)
    save_json({'dataset':args.dataset,'counts':{k:len(v) for k,v in splits.items()},'train_ids':[x.sample_id for x in splits['train']], 'dev_ids':[x.sample_id for x in splits['dev']], 'test_ids':[x.sample_id for x in splits['test']], 'gloss_vocab':gv.to_dict(),'text_vocab':tv.to_dict(),'config':cfg.to_dict()},out/'manifest.json')
    print(json.dumps({'dataset':args.dataset,'counts':{k:len(v) for k,v in splits.items()},'gloss_vocab':len(gv),'text_vocab':len(tv)},indent=2))
if __name__=='__main__': main()
