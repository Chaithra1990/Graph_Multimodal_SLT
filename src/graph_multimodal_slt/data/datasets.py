from __future__ import annotations
import torch
from torch.utils.data import Dataset
from .annotations import read_split, Sample
from .features import load_feature_file
from .vocab import Vocab

class SignLanguageDataset(Dataset):
    def __init__(self, samples:list[Sample], feature_root, max_seq_len, gloss_vocab, text_vocab, rgb_dim=2048, pose_joints=25, face_dim=136):
        self.samples=samples; self.feature_root=feature_root; self.max_seq_len=max_seq_len
        self.gloss_vocab=gloss_vocab; self.text_vocab=text_vocab
        self.rgb_dim=rgb_dim; self.pose_joints=pose_joints; self.face_dim=face_dim
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s=self.samples[idx]
        feat=load_feature_file(self.feature_root,s.sample_id,self.max_seq_len,self.rgb_dim,self.pose_joints,self.face_dim)
        gloss=torch.tensor(self.gloss_vocab.encode(s.gloss_tokens),dtype=torch.long)
        text_in=torch.tensor(self.text_vocab.encode(s.text_tokens,add_bos=True),dtype=torch.long)
        text_tgt=torch.tensor(self.text_vocab.encode(s.text_tokens,add_eos=True),dtype=torch.long)
        return {**feat,"id":s.sample_id,"gloss":gloss,"text_input":text_in,"text_target":text_tgt,"reference":s.text,"gloss_pad_idx":self.gloss_vocab.pad_idx,"text_pad_idx":self.text_vocab.pad_idx}

def build_vocabs(train_samples):
    return Vocab.build([s.gloss_tokens for s in train_samples]), Vocab.build([s.text_tokens for s in train_samples])

def make_splits(dataset, data_root):
    from pathlib import Path
    root=Path(data_root); ann=root/"annotations"; videos=root/"videos"
    if dataset=="phoenix2014t": names={k:ann/f"PHOENIX-2014-T.{k}.corpus.csv" for k in ("train","dev","test")}
    else: names={k:ann/f"csl_daily.{k}.json" for k in ("train","dev","test")}
    return {k:read_split(dataset,p,videos) for k,p in names.items()}

def slt_collate_fn(batch):
    B=len(batch); T=max(x["length"] for x in batch)
    D=batch[0]["rgb"].shape[1]; V=batch[0]["pose"].shape[1]; F=batch[0]["face"].shape[1]
    rgb=torch.zeros(B,T,D); pose=torch.zeros(B,T,V,2); face=torch.zeros(B,T,F); lengths=torch.tensor([x["length"] for x in batch],dtype=torch.long)
    for i,x in enumerate(batch):
        t=x["length"]; rgb[i,:t]=x["rgb"]; pose[i,:t]=x["pose"]; face[i,:t]=x["face"]
    def pad_1d(key,pad):
        m=max(len(x[key]) for x in batch); out=torch.full((B,m),pad,dtype=torch.long)
        for i,x in enumerate(batch): out[i,:len(x[key])]=x[key]
        return out
    gloss_lens=torch.tensor([len(x["gloss"]) for x in batch],dtype=torch.long)
    return {"rgb":rgb,"pose":pose,"face":face,"lengths":lengths,"gloss":pad_1d("gloss",batch[0]["gloss_pad_idx"]),"gloss_lens":gloss_lens,"text_input":pad_1d("text_input",0),"text_target":pad_1d("text_target",0),"ids":[x["id"] for x in batch],"references":[x["reference"] for x in batch]}
