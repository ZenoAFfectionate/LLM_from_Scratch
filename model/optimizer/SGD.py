import math
import torch
from typing import Optional, Callable, Iterable


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
    

if __name__ == "__main__":
    print("="*60)
    print("Testing SGD Optimizer")
    print("="*60)
    

    print("\n\n[Test 1] Multi-step update verification")
    print("-"*60)

    weights = torch.nn.Parameter(torch.tensor([10.0]))
    optimizer = SGD([weights], lr=1.0)

    print(f"Initial weight: {weights.item():.6f}")

    for step in range(3):
        optimizer.zero_grad()

        # Loss: L = w^2, Gradient: dL/dw = 2*w
        loss = weights ** 2
        loss.backward()

        grad = weights.grad.item()
        old_weight = weights.item()

        # Calculate expected update
        # w_new = w_old - (lr / sqrt(step+1)) * grad
        lr_adjusted = 1.0 / math.sqrt(step + 1)
        expected_weight = old_weight - lr_adjusted * grad

        optimizer.step()

        print(f"Step {step}: w={old_weight:.6f}, grad={grad:.6f}, "
              f"lr_eff={lr_adjusted:.6f}, w_new={weights.item():.6f}, "
              f"expected={expected_weight:.6f}, match={abs(weights.item() - expected_weight) < 1e-5}")


    print("\n[Test 2] Convergence test (minimize w^2)")
    print("-"*60)

    weights = torch.nn.Parameter(torch.tensor([5.0]))
    optimizer = SGD([weights], lr=0.5)

    print(f"Initial: w={weights.item():.4f}, loss={(weights**2).item():.4f}")

    for i in range(10):
        optimizer.zero_grad()
        loss = (weights ** 2)
        loss.backward()
        optimizer.step()

        if i % 3 == 0 or i == 9:
            print(f"Step {i+1}: w={weights.item():.4f}, loss={loss.item():.4f}")

    print(f"\nFinal weight: {weights.item():.6f} (should be close to 0)")
    print(f"Converged: {abs(weights.item()) < 0.1}")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)