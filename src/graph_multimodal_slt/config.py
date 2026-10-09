from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import yaml

@dataclass
class Config:
    dataset: str
    max_seq_len: int = 300
    rgb_dim: int = 2048
    pose_joints: int = 25
    pose_channels: int = 2
    face_dim: int = 136
    d_model: int = 512
    d_ff: int = 2048
    nhead: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    dropout: float = 0.1
    fusion_gamma: float = 0.5
    batch_size: int = 16
    grad_accum_steps: int = 2
    epochs: int = 100
    lr_max: float = 3e-4
    warmup_steps: int = 4000
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.998
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    lambda_ctc_start: float = 1.0
    lambda_ctc_end: float = 0.3
    seeds: tuple[int, ...] = (42, 123, 456)
    num_workers: int = 4
    max_decode_len: int = 80
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    rgb_backbone: str = "resnet50"
    rgb_weights: str = "DEFAULT"
    frame_size: int = 224
    frame_stride: int = 1
    openpose_body25: bool = True
    face_landmarks: int = 68

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        raw["seeds"] = tuple(raw.get("seeds", (42, 123, 456)))
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        return d
