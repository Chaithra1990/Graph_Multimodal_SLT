from __future__ import annotations
from pathlib import Path
import json, math, time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from ..evaluation.metrics import evaluate_all
from ..models.architecture import LabelSmoothingCrossEntropy
from ..utils.checkpoint import save_checkpoint
from ..utils.io import save_json

class Trainer:
    def __init__(self,model,train_loader,dev_loader,test_loader,gloss_vocab,text_vocab,cfg,run_dir,device):
        self.model=model; self.train_loader=train_loader; self.dev_loader=dev_loader; self.test_loader=test_loader; self.gv=gloss_vocab; self.tv=text_vocab; self.cfg=cfg; self.run_dir=Path(run_dir); self.device=device; self.run_dir.mkdir(parents=True,exist_ok=True)
        self.ctc=nn.CTCLoss(blank=gloss_vocab.pad_idx,zero_infinity=True); self.ce=LabelSmoothingCrossEntropy(cfg.label_smoothing,text_vocab.pad_idx)
        self.opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr_max,betas=(cfg.adam_beta1,cfg.adam_beta2),weight_decay=cfg.weight_decay)
        from .scheduler import WarmupInverseSqrtScheduler
        self.scheduler=WarmupInverseSqrtScheduler(self.opt,cfg.warmup_steps,cfg.lr_max)
        self.scaler=GradScaler(enabled=device.type=='cuda'); self.best=-1.0
    def _move(self,b): return {k:(v.to(self.device) if torch.is_tensor(v) else v) for k,v in b.items()}
    def _lambda_ctc(self,epoch):
        p=(epoch-1)/max(1,self.cfg.epochs-1); return self.cfg.lambda_ctc_start+(self.cfg.lambda_ctc_end-self.cfg.lambda_ctc_start)*p
    def train_epoch(self,epoch):
        self.model.train(); total=0.; self.opt.zero_grad(set_to_none=True); start=time.time(); lam=self._lambda_ctc(epoch)
        for step,b in enumerate(tqdm(self.train_loader,desc=f'Epoch {epoch}',leave=False)):
            b=self._move(b)
            with autocast(device_type='cuda',enabled=self.device.type=='cuda'):
                gloss_logits,text_logits,_,_=self.model(b['rgb'],b['pose'],b['face'],b['lengths'],b['text_input'])
                lctc=self.ctc(gloss_logits.log_softmax(-1).transpose(0,1),b['gloss'],b['lengths'],b['gloss_lens'])
                lce=self.ce(text_logits.reshape(-1,text_logits.size(-1)),b['text_target'].reshape(-1)); loss=(lam*lctc+lce)/self.cfg.grad_accum_steps
            self.scaler.scale(loss).backward()
            if (step+1)%self.cfg.grad_accum_steps==0 or step==len(self.train_loader)-1:
                self.scaler.unscale_(self.opt); torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.max_grad_norm); self.scaler.step(self.opt); self.scaler.update(); self.scheduler.step(); self.opt.zero_grad(set_to_none=True)
            total+=loss.item()*self.cfg.grad_accum_steps
        return {'loss':total/max(1,len(self.train_loader)),'lambda_ctc':lam,'seconds':time.time()-start,'lr':self.opt.param_groups[0]['lr']}
    @torch.no_grad()
    def predict(self,loader):
        self.model.eval(); refs=[]; hyps=[]
        for b in tqdm(loader,desc='Evaluate',leave=False):
            b=self._move(b); ids=self.model.greedy_decode(b['rgb'],b['pose'],b['face'],b['lengths'],self.tv.bos_idx,self.tv.eos_idx,self.cfg.max_decode_len)
            refs.extend(b['references']); hyps.extend([self.tv.decode(x.tolist()) for x in ids])
        return refs,hyps
    def evaluate(self,loader):
        r,h=self.predict(loader); return evaluate_all(r,h),r,h
    def fit(self):
        history=[]
        for epoch in range(1,self.cfg.epochs+1):
            tr=self.train_epoch(epoch); metrics,_,_=self.evaluate(self.dev_loader); rec={'epoch':epoch,**tr,**metrics}; history.append(rec); save_json(history,self.run_dir/'history.json')
            if metrics['BLEU-4']>self.best:
                self.best=metrics['BLEU-4']; save_checkpoint(self.run_dir/'best.pt',self.model,self.opt,self.scheduler,self.scaler,epoch,self.best,self.cfg.to_dict(),{'gloss':self.gv.to_dict(),'text':self.tv.to_dict()})
                save_json({'epoch':epoch,'metrics':metrics},self.run_dir/'best_metrics.json')
        ck=torch.load(self.run_dir/'best.pt',map_location=self.device,weights_only=False); self.model.load_state_dict(ck['model']); test_metrics,refs,hyps=self.evaluate(self.test_loader); save_json({'metrics':test_metrics,'references':refs,'hypotheses':hyps},self.run_dir/'test_results.json'); return test_metrics
