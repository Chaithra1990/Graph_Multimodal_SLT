from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv, json
from typing import Iterable

@dataclass(frozen=True)
class Sample:
    sample_id: str
    gloss: str
    text: str
    video: str | None = None

    @property
    def gloss_tokens(self): return self.gloss.strip().split()
    @property
    def text_tokens(self): return self.text.strip().split()

def _find_video(video_root: Path, sample_id: str) -> Path:
    candidates = [video_root / sample_id]
    for ext in (".mp4", ".avi", ".webm", ".mov", ".mkv"):
        candidates.append(video_root / f"{sample_id}{ext}")
    for p in candidates:
        if p.is_file(): return p
    matches=list(video_root.rglob(sample_id)) + [p for ext in ("*.mp4","*.avi","*.webm","*.mov","*.mkv") for p in video_root.rglob(ext) if p.stem==sample_id]
    if matches: return matches[0]
    raise FileNotFoundError(f"No real video found for sample '{sample_id}' below {video_root}")

def read_phoenix_corpus(path: str | Path, video_root: str | Path | None = None) -> list[Sample]:
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f"Missing PHOENIX corpus: {p}")
    rows=[]
    with p.open(encoding="utf-8") as f:
        reader=csv.reader(f, delimiter="|")
        header=next(reader, None)
        if not header: raise ValueError(f"Empty PHOENIX corpus: {p}")
        for row in reader:
            if len(row)<3: continue
            sid, gloss, text=row[0].strip(), row[1].strip(), row[2].strip()
            if sid.lower() in {"name","id"}: continue
            video=str(_find_video(Path(video_root), sid)) if video_root else None
            rows.append(Sample(sid, gloss, text, video))
    if not rows: raise ValueError(f"No samples parsed from {p}")
    return rows

def _json_items(data):
    if isinstance(data,list): return data
    if isinstance(data,dict):
        for key in ("samples","data","annotations","items"):
            if isinstance(data.get(key),list): return data[key]
        return list(data.values())
    raise TypeError("Unsupported JSON annotation structure")

def read_csldaily_json(path: str | Path, video_root: str | Path | None = None) -> list[Sample]:
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f"Missing CSL-Daily annotation: {p}")
    data=json.loads(p.read_text(encoding="utf-8"))
    out=[]
    for item in _json_items(data):
        if not isinstance(item,dict): continue
        sid=item.get("name") or item.get("file") or item.get("id") or item.get("video_id")
        gloss=item.get("label_gloss") or item.get("gloss") or item.get("glosses")
        text=item.get("label_text") or item.get("text") or item.get("translation")
        if sid is None or gloss is None or text is None: continue
        if isinstance(gloss,list): gloss=" ".join(map(str,gloss))
        if isinstance(text,list): text=" ".join(map(str,text))
        video=str(_find_video(Path(video_root), str(sid))) if video_root else None
        out.append(Sample(str(sid),str(gloss),str(text),video))
    if not out: raise ValueError(f"No valid samples parsed from {p}")
    return out

def read_split(dataset, annotation_path, video_root=None):
    return read_phoenix_corpus(annotation_path,video_root) if dataset=="phoenix2014t" else read_csldaily_json(annotation_path,video_root)

def validate_official_counts(dataset, splits):
    expected={"phoenix2014t": {"train":7096,"dev":519,"test":642}, "csldaily":{"train":18401,"dev":1077,"test":1176}}
    for split, samples in splits.items():
        n=len(samples)
        exp=expected[dataset][split]
        if n != exp:
            raise ValueError(f"{dataset} {split} has {n} samples; manuscript reports official count {exp}. Refusing to proceed because this would not reproduce the reported split.")
