from __future__ import annotations
from collections import Counter
import json

class Vocab:
    def __init__(self, tokens=None, specials=("<pad>", "<unk>", "<bos>", "<eos>")):
        self.specials = list(specials)
        self.itos = list(specials)
        if tokens:
            for t in tokens:
                if t not in self.itos:
                    self.itos.append(t)
        self.stoi = {t:i for i,t in enumerate(self.itos)}

    @classmethod
    def build(cls, sequences, specials=("<pad>", "<unk>", "<bos>", "<eos>")):
        c = Counter(tok for seq in sequences for tok in seq)
        return cls([x for x,_ in sorted(c.items(), key=lambda kv:(-kv[1], kv[0]))], specials)

    @property
    def pad_idx(self): return self.stoi["<pad>"]
    @property
    def unk_idx(self): return self.stoi["<unk>"]
    @property
    def bos_idx(self): return self.stoi["<bos>"]
    @property
    def eos_idx(self): return self.stoi["<eos>"]
    def __len__(self): return len(self.itos)
    def encode(self, tokens, add_bos=False, add_eos=False):
        out = [self.stoi.get(t, self.unk_idx) for t in tokens]
        if add_bos: out.insert(0, self.bos_idx)
        if add_eos: out.append(self.eos_idx)
        return out
    def decode(self, ids, strip_special=True):
        out=[]
        for i in ids:
            t=self.itos[int(i)] if 0 <= int(i) < len(self.itos) else "<unk>"
            if strip_special and t in self.specials: continue
            out.append(t)
        return " ".join(out)
    def to_dict(self): return {"itos": self.itos, "specials": self.specials}
    @classmethod
    def from_dict(cls, d):
        v=cls([], d.get("specials", ["<pad>","<unk>","<bos>","<eos>"]))
        v.itos=d["itos"]; v.stoi={t:i for i,t in enumerate(v.itos)}; return v
