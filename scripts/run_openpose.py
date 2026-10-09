#!/usr/bin/env python3
"""Run a real OpenPose BODY_25 + face extraction over dataset videos.

OpenPose itself is not redistributed. Install a compatible OpenPose release and
pass its executable/model directory. The script never creates substitute
keypoints when OpenPose fails.
"""
import argparse, subprocess
from pathlib import Path
from graph_multimodal_slt.data.datasets import make_splits

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=['phoenix2014t','csldaily']); ap.add_argument('--data-root',required=True); ap.add_argument('--openpose-bin',required=True); ap.add_argument('--model-folder',required=True); ap.add_argument('--output-root',required=True); ap.add_argument('--split',choices=['train','dev','test']); args=ap.parse_args()
    splits=make_splits(args.dataset,args.data_root); names=[args.split] if args.split else ['train','dev','test']; out=Path(args.output_root); out.mkdir(parents=True,exist_ok=True)
    for split in names:
        for sample in splits[split]:
            if not sample.video: raise RuntimeError(f'No video resolved for {sample.sample_id}')
            target=out/sample.sample_id; target.mkdir(parents=True,exist_ok=True)
            cmd=[args.openpose_bin,'--video',sample.video,'--write_json',str(target),'--display','0','--render_pose','0','--model_pose','BODY_25','--face','--hand','--model_folder',args.model_folder]
            print('Running:', ' '.join(map(str,cmd)))
            subprocess.run(cmd,check=True)
if __name__=='__main__': main()
