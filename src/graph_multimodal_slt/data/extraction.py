from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

BODY25_SHOULDERS=(2,5)

class RGBExtractor:
    def __init__(self, device='cuda', image_size=224, weights='DEFAULT'):
        self.device=torch.device(device if device=='cpu' or torch.cuda.is_available() else 'cpu')
        w=ResNet50_Weights.DEFAULT if weights=='DEFAULT' else ResNet50_Weights.DEFAULT
        m=resnet50(weights=w); m.fc=torch.nn.Identity(); self.model=m.eval().to(self.device); self.transform=w.transforms()
        self.image_size=image_size
    @torch.inference_mode()
    def extract_video(self, video_path, stride=1):
        cap=cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise RuntimeError(f'Cannot open video: {video_path}')
        frames=[]; i=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if i%stride==0:
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); frames.append(self.transform(Image.fromarray(frame)))
            i+=1
        cap.release()
        if not frames: raise RuntimeError(f'Video has no readable frames: {video_path}')
        feats=[]
        for j in range(0,len(frames),64):
            x=torch.stack(frames[j:j+64]).to(self.device); feats.append(self.model(x).cpu())
        return torch.cat(feats,0)

def _load_jsons(sample_dir):
    p=Path(sample_dir)
    if not p.is_dir(): raise FileNotFoundError(f'Missing OpenPose JSON directory: {p}')
    files=sorted(p.glob('*.json'))
    if not files: raise FileNotFoundError(f'No OpenPose JSON files in {p}')
    return files

def _single_person(frame_json, sample_id):
    people=frame_json.get('people',[])
    if not people: raise ValueError(f'No detected person in OpenPose frame for {sample_id}')
    # Deterministic primary-person rule: highest BODY_25 mean confidence.
    def score(p):
        k=np.asarray(p.get('pose_keypoints_2d',[]),dtype=np.float32).reshape(-1,3)
        return float(k[:,2].mean()) if len(k) else -1
    return max(people,key=score)

def parse_openpose(sample_dir, sample_id, expected_frames=None, stride=1):
    files=_load_jsons(sample_dir)[::stride]
    pose=[]; face=[]
    for f in files:
        d=json.loads(f.read_text(encoding='utf-8')); p=_single_person(d,sample_id)
        body=np.asarray(p.get('pose_keypoints_2d',[]),dtype=np.float32).reshape(-1,3)
        if body.shape[0] < 25: raise ValueError(f'{f}: expected OpenPose BODY_25, got {body.shape[0]} joints')
        xy=body[:25,:2]; conf=body[:25,2]
        if np.any(conf<=0): raise ValueError(f'{f}: BODY_25 contains undetected joints; preprocessing cannot silently invent coordinates')
        face_raw=np.asarray(p.get('face_keypoints_2d',[]),dtype=np.float32).reshape(-1,3)
        if face_raw.shape[0] < 68: raise ValueError(f'{f}: expected at least 68 facial landmarks, got {face_raw.shape[0]}')
        fxy=face_raw[:68,:2]; fconf=face_raw[:68,2]
        if np.any(fconf<=0): raise ValueError(f'{f}: facial landmarks contain undetected points')
        rs,ls=xy[BODY25_SHOULDERS[0]],xy[BODY25_SHOULDERS[1]]
        center=(rs+ls)/2; scale=np.linalg.norm(rs-ls)
        if scale<=1e-6: raise ValueError(f'{f}: invalid shoulder distance')
        pose.append((xy-center)/scale); face.append(fxy)
    pose=torch.tensor(np.stack(pose),dtype=torch.float32); face=torch.tensor(np.stack(face),dtype=torch.float32)
    if expected_frames is not None and len(pose)!=expected_frames: raise ValueError(f'{sample_id}: RGB frames={expected_frames}, OpenPose frames={len(pose)}. They must be synchronized.')
    # Normalize facial coordinates to image-independent coordinates using the same shoulder frame.
    # We use the first two BODY_25 shoulder points from each corresponding frame.
    # Face is returned in pixel coordinates here; caller normalizes using pose metadata.
    return pose,face,files

def normalize_face(face_xy, pose_dir, sample_id):
    files=_load_jsons(pose_dir); out=[]
    for i,f in enumerate(files):
        d=json.loads(f.read_text(encoding='utf-8')); p=_single_person(d,sample_id); body=np.asarray(p['pose_keypoints_2d'],dtype=np.float32).reshape(-1,3); rs,ls=body[2,:2],body[5,:2]; center=(rs+ls)/2; scale=np.linalg.norm(rs-ls); out.append((face_xy[i].numpy()-center)/max(scale,1e-6))
    return torch.tensor(np.stack(out),dtype=torch.float32).reshape(len(out),136)

def extract_sample(video_path, openpose_dir, output_path, extractor, sample_id, stride=1):
    rgb=extractor.extract_video(video_path,stride)
    pose,face,_=parse_openpose(openpose_dir,sample_id,expected_frames=rgb.shape[0],stride=stride)
    face=normalize_face(face,openpose_dir,sample_id)[::stride]
    if not (rgb.shape[0]==pose.shape[0]==face.shape[0]): raise RuntimeError(f'{sample_id}: modality frame counts do not match')
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'rgb':rgb.half().contiguous(),'pose':pose.contiguous(),'face':face.contiguous(),'sample_id':sample_id},output_path)
