from pathlib import Path
import json, yaml, torch

def require_file(path: str | Path, description: str = "required file") -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing {description}: {p}")
    return p

def require_dir(path: str | Path, description: str = "required directory") -> Path:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"Missing {description}: {p}")
    return p

def save_json(obj, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_json(path):
    return json.loads(require_file(path).read_text(encoding="utf-8"))

def save_yaml(obj, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")

def atomic_torch_save(obj, path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(p)
