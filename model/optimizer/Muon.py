import torch

# define coefficients here:
a = +3.4445
b = -4.7750
c = +2.0315


def newtonschulz5_orthogonalization(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval.
    """
    assert G.ndim >= 2, "Input tensor must be at least 2D"
    X = G.to(torch.bfloat16)

    if G.size(-2) > G.size(-1): X = X.mT  # transpose for tall matrices

    # ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy
        X = a * X + B @ X

    if G.size(-2) > G.size(-1): X = X.mT  # transpose for tall matrices

    return X


def muon_update(grad, momentum, scaling_factor, beta=0.95, ns_steps=5, nesterov=True):
    """
    Compute Muon update with momentum and orthogonalization.

    Args:
        grad: Gradient tensor
        momentum: Momentum buffer
        beta: Momentum coefficient (default: 0.95)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        nesterov: Whether to use Nesterov momentum (default: True)

    Returns:
        Orthogonalized update tensor
    """
    # update momentum with EMA
    momentum.lerp_(grad, 1 - beta)
    # Use non-in-place lerp to avoid modifying the gradient tensor
    update = grad.lerp(momentum, beta) if nesterov else momentum
    # for the case of conv filters
    if update.ndim == 4: 
        update = update.view(len(update), -1)
    # perform Newton-Schulz orthogonalization
    update = newtonschulz5_orthogonalization(update, steps=ns_steps)
    return update * scaling_factor


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz (This implementation contain aux Adam)

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.

    Arguments:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.02) - in units of spectral norm per update
        weight_decay: Weight decay coefficient (default: 0) - AdamW-style weight decay
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step. """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    # initialize momentum buffer
                    state["momentum_buffer"] = torch.zeros_like(p)
                    # compute and cache scaling factor
                    if p.ndim >= 2:
                        m, n = p.shape[-2], p.shape[-1]
                        scale_value = max(1.0, m / n) ** 0.5
                        state["scaling_factor"] = scale_value
                    else:
                        state["scaling_factor"] = 1.0

                # compute Muon update with orthogonalization
                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],  # 
                    state["scaling_factor"],   # 
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                    nesterov=group["nesterov"]
                )

                # apply AdamW-style weight decay and update parameters
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


if __name__ == "__main__":
    print("="*60)
    print("Testing Muon Optimizer")
    print("="*60)


    print("\n\n[Test 1] Multi-step update verification (2D weights)")
    print("-"*60)

    # Muon works best with 2D matrix parameters
    weights = torch.nn.Parameter(torch.tensor([[10.0, 8.0], [6.0, 4.0]]))
    optimizer = Muon([weights], lr=0.01, momentum=0.95, weight_decay=0.0)

    print(f"Initial weights:\n{weights.data}")

    for step in range(3):
        optimizer.zero_grad()

        # Loss: L = sum(w^2), Gradient: dL/dw = 2*w
        loss = (weights ** 2).sum()
        loss.backward()

        old_weight_norm = weights.data.norm().item()
        grad_norm = weights.grad.norm().item()

        optimizer.step()

        print(f"Step {step}: weight_norm={old_weight_norm:.6f}, grad_norm={grad_norm:.6f}, "
              f"new_weight_norm={weights.data.norm().item():.6f}")


    print("\n[Test 2] Convergence test (minimize sum(w^2) for 2D matrix)")
    print("-"*60)

    weights = torch.nn.Parameter(torch.tensor([[5.0, 4.0], [3.0, 2.0]]))
    optimizer = Muon([weights], lr=0.08, momentum=0.95, weight_decay=0.0)

    initial_loss = (weights ** 2).sum().item()
    initial_norm = weights.norm().item()
    print(f"Initial: weight_norm={initial_norm:.4f}, loss={initial_loss:.4f}")

    for i in range(25):
        optimizer.zero_grad()
        loss = (weights ** 2).sum()
        loss.backward()
        optimizer.step()

        if i % 6 == 0 or i == 24:
            print(f"Step {i+1}: weight_norm={weights.norm().item():.4f}, loss={loss.item():.4f}")

    final_norm = weights.norm().item()
    print(f"\nFinal weight norm: {final_norm:.6f} (should be smaller than initial {initial_norm:.4f})")
    print(f"Loss reduced: {loss.item() < initial_loss}")
    print(f"Converged: {final_norm < initial_norm}")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)