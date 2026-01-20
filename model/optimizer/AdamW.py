import torch
from typing import Optional, Callable, Iterable


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
                    # CRITICAL FIX: Optimizer states must ALWAYS be FP32 for numerical stability
                    # Even when using BF16 weights, optimizer momentum and variance need FP32 precision
                    state['m'] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, dtype=torch.float32, memory_format=torch.preserve_format)

                # CRITICAL FIX: Cast gradient to FP32 for optimizer updates
                grad = p.grad.float()  # Always use FP32 gradients in optimizer
                m, v = state['m'], state['v']
                state["step"] += 1
                t = state['step']
                # update both 1st moment estimator and 2nd moment estimator (in FP32)
                m.mul_(beta1).add_(grad**1, alpha=1 - beta1)  # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t^1
                v.mul_(beta2).add_(grad**2, alpha=1 - beta2)  # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                # compute adjusted alpha for iteration t
                step_size = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)
                # CRITICAL FIX: Compute update in FP32, then convert to parameter dtype
                update = (m / (v.sqrt() + eps)).to(dtype=p.dtype)
                p.add_(update, alpha=-step_size)
                if weight_decay != 0: p.add_(p, alpha=-lr * weight_decay)

        return loss


if __name__ == "__main__":
    print("="*60)
    print("Testing AdamW Optimizer")
    print("="*60)


    print("\n\n[Test 1] Multi-step update verification")
    print("-"*60)

    weights = torch.nn.Parameter(torch.tensor([10.0]))
    optimizer = AdamW([weights], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    print(f"Initial weight: {weights.item():.6f}")

    for step in range(3):
        optimizer.zero_grad()

        # Loss: L = w^2, Gradient: dL/dw = 2*w
        loss = weights ** 2
        loss.backward()

        grad = weights.grad.item()
        old_weight = weights.item()

        optimizer.step()

        print(f"Step {step}: w={old_weight:.6f}, grad={grad:.6f}, "
              f"w_new={weights.item():.6f}")


    print("\n[Test 2] Convergence test (minimize w^2)")
    print("-"*60)

    weights = torch.nn.Parameter(torch.tensor([5.0]))
    optimizer = AdamW([weights], lr=0.3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    print(f"Initial: w={weights.item():.4f}, loss={(weights**2).item():.4f}")

    for i in range(15):
        optimizer.zero_grad()
        loss = (weights ** 2)
        loss.backward()
        optimizer.step()

        if i % 4 == 0 or i == 14:
            print(f"Step {i+1}: w={weights.item():.4f}, loss={loss.item():.4f}")

    print(f"\nFinal weight: {weights.item():.6f} (should be smaller than initial)")
    print(f"Loss reduced: {(weights**2).item() < 25.0}")
    print(f"Converged: {abs(weights.item()) < 1.0}")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)