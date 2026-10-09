import math
from torch.optim.lr_scheduler import _LRScheduler
class WarmupInverseSqrtScheduler(_LRScheduler):
    def __init__(self,optimizer,warmup_steps,base_lr,last_epoch=-1):
        self.warmup_steps=warmup_steps; self.base_lr=base_lr; super().__init__(optimizer,last_epoch)
    def get_lr(self):
        step=max(1,self.last_epoch+1); scale=(step/self.warmup_steps) if step<=self.warmup_steps else math.sqrt(self.warmup_steps/step); return [self.base_lr*scale for _ in self.optimizer.param_groups]
