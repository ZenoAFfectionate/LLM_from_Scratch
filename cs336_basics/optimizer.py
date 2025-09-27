import math
import torch
import torch.nn as nn

from typing import Optional
from collections.abc import Callable, Iterable


# -------------------------------------------
#  Problem 12: Implement SGD Optimizer Class
# -------------------------------------------
class SGD(torch.optim.Optimizer):
    """ PyTorch implementation of Stochastic Gradient Descent """
    def __init__(self, params, lr=1e-3):
        if lr < 0: raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """ make one update of the parameters """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None: continue
                # get the gradient and iteration number
                state = self.state[p]  # get state associated with p
                t = state.get("t", 0)  # get iteration number t
                grad = p.grad          # get the gradient of loss
                # perform the in-place parameter update
                p.add_(grad, alpha=-(lr / math.sqrt(t + 1)))
                state["t"] = t + 1

        return loss


# ---------------------------------------------
#  Problem 13: Implement AdamW Optimizer Class
# ---------------------------------------------
class AdamW(torch.optim.Optimizer):
    """ PyTorch implementation of AdamW optimizer """
    def __init__(self, 
                 params: Iterable[torch.nn.Parameter], 
                 lr: float = 1e-3, 
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, 
                 weight_decay: float = 1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        # pack all hyper-parameters into the default dictionary
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """ make one update of the parameters """
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr, eps = group['lr'], group['eps']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None: continue

                state = self.state[p]
                if len(state) == 0:  # lazy allocation
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                grad = p.grad  # get the gradient of the parameter
                m, v = state['m'], state['v']
                state["step"] += 1
                t = state['step']
                # update both 1st moment estimator and 2nd moment estimator
                m.mul_(beta1).add_(grad**1, alpha=1 - beta1)  # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t^1
                v.mul_(beta2).add_(grad**2, alpha=1 - beta2)  # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                # compute adjusted alpha for iteration t
                step_size = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)
                # update parameter and apply weight decay
                p.add_(m / (v.sqrt() + eps), alpha=-step_size)
                if weight_decay != 0: p.add_(p, alpha=-lr * weight_decay)

        return loss


# ------------------------------------------------------------------
#  Problem 14: Implement Cosine learning rate scheduler with warmup
# ------------------------------------------------------------------
def cos_learning_rate_schedule_with_warmup(t: int, max_lr: float, min_lr: float, warmup_iter: int, cos_iter: int):
    """  """
    # step 1: linear warmup phase
    if t < warmup_iter:
        return max_lr * (t / warmup_iter)
    # step 2: cosine annealing
    elif t < cos_iter:
        factor = (t - warmup_iter) / (cos_iter - warmup_iter)
        return min_lr + 0.5 * (1 + math.cos(factor * 3.1415926)) * (max_lr - min_lr)
    # step 3: post annealing
    else:
        return min_lr



if __name__ == "__main__":
    # a simple test example SGD 

    for lr in [1e1, 1e2, 1e3]:
        print(f"> Under the learning rate {lr}:")
        # initialize weights and optimizer
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=1)

        for t in range(100):
            # calculate gradient and loss
            opt.zero_grad()
            loss = (weights**2).mean()

            print(f'   loss at epoch {t} is:', loss.cpu().item())

            # update weights and optimizer
            loss.backward()
            opt.step()
        
        print()
