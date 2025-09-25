# utils.py
import os, random, time, math
from dataclasses import dataclass

@dataclass
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.05
    decay_steps: int = 200_000

    def __call__(self, step: int) -> float:
        t = min(1.0, step / max(1, self.decay_steps))
        return self.start + (self.end - self.start) * t

def linear_lr_schedule(initial_lr: float, final_lr: float, total_steps: int):
    def f(step: int):
        t = min(1.0, step / max(1, total_steps))
        return initial_lr + (final_lr - initial_lr) * t
    return f

def seed_everything(seed: int):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AverageMeter:
    def __init__(self): self.v = 0.0; self.n=0
    def update(self, x, k=1): self.v = (self.v*self.n + x*k)/(self.n+k); self.n+=k
