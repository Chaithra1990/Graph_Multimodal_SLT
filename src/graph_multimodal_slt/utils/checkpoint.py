from pathlib import Path
import torch
from .io import atomic_torch_save

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_bleu, config, vocabularies):
    atomic_torch_save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_bleu": best_bleu,
        "config": config,
        "vocabularies": vocabularies,
    }, path)

def load_checkpoint(path, model, device, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
    return ckpt
