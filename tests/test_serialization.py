import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import get_adamw_cls, run_load_checkpoint, run_save_checkpoint


class _TestNet(nn.Module):
    def __init__(self, d_input: int = 100, d_output: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(d_input, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, d_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def are_optimizers_equal(optimizer1_state_dict, optimizer2_state_dict, atol=1e-8, rtol=1e-5):
    # Check if the keys of the main dictionaries are equal (e.g., 'state', 'param_groups')
    if set(optimizer1_state_dict.keys()) != set(optimizer2_state_dict.keys()):
        return False

    # Check parameter groups are identical
    if optimizer1_state_dict["param_groups"] != optimizer2_state_dict["param_groups"]:
        return False

    # Check states
    state1 = optimizer1_state_dict["state"]
    state2 = optimizer2_state_dict["state"]
    if set(state1.keys()) != set(state2.keys()):
        return False

    for key in state1:
        # Assuming state contents are also dictionaries
        if set(state1[key].keys()) != set(state2[key].keys()):
            return False

        for sub_key in state1[key]:
            item1 = state1[key][sub_key]
            item2 = state2[key][sub_key]

            # If both items are tensors, use torch.allclose
            if torch.is_tensor(item1) and torch.is_tensor(item2):
                if not torch.allclose(item1, item2, atol=atol, rtol=rtol):
                    return False
            # For non-tensor items, check for direct equality
            elif item1 != item2:
                return False
    return True


def test_checkpointing(tmp_path):
    torch.manual_seed(42)
    d_input = 100
    d_output = 10
    num_iters = 10

    model = _TestNet(d_input=d_input, d_output=d_output)
    optimizer = get_adamw_cls()(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # Use 1000 optimization steps for testing
    it = 0
    for _ in range(num_iters):
        optimizer.zero_grad()
        x = torch.rand(d_input)
        y = torch.rand(d_output)
        y_hat = model(x)
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        optimizer.step()
        it += 1

    serialization_path = tmp_path / "checkpoint.pt"
    # Save the model
    run_save_checkpoint(
        model,
        optimizer,
        iteration=it,
        out=serialization_path,
    )

    # Load the model back again
    new_model = _TestNet(d_input=d_input, d_output=d_output)
    new_optimizer = get_adamw_cls()(
        new_model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    loaded_iterations = run_load_checkpoint(src=serialization_path, model=new_model, optimizer=new_optimizer)
    assert it == loaded_iterations

    # Compare the loaded model state with the original model state
    original_model_state = model.state_dict()
    original_optimizer_state = optimizer.state_dict()
    new_model_state = new_model.state_dict()
    new_optimizer_state = new_optimizer.state_dict()

    # Check that state dict keys match
    assert set(original_model_state.keys()) == set(new_model_state.keys())
    assert set(original_optimizer_state.keys()) == set(new_optimizer_state.keys())

    # compare the model state dicts
    for key in original_model_state.keys():
        numpy.testing.assert_allclose(
            original_model_state[key].detach().numpy(),
            new_model_state[key].detach().numpy(),
        )
    # compare the optimizer state dicts
    assert are_optimizers_equal(original_optimizer_state, new_optimizer_state)


# =============================================================================
# Additional checkpoint behaviour tests (direct, not via adapter shim).
# =============================================================================

def _make_tiny_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_checkpoint_auto_appends_pt_suffix(tmp_path):
    from model.utils import save_checkpoint
    model = _make_tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    out = tmp_path / "no_suffix"
    save_checkpoint(model, opt, iteration=1, out=str(out))
    assert (tmp_path / "no_suffix.pt").exists()


def test_checkpoint_handles_orig_mod_prefix(tmp_path):
    """state-dict keys with `_orig_mod.` prefix (torch.compile) must be stripped on load."""
    from model.utils import load_checkpoint
    model = _make_tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ckpt_path = tmp_path / "compiled.pt"
    sd = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    torch.save({
        "model_state_dict": sd,
        "optimizer_state_dict": opt.state_dict(),
        "iteration": 7,
    }, ckpt_path)
    fresh = _make_tiny_model()
    it = load_checkpoint(str(ckpt_path), fresh, opt)
    assert it == 7
    for p1, p2 in zip(model.parameters(), fresh.parameters()):
        torch.testing.assert_close(p1.data, p2.data)


def test_checkpoint_roundtrip_preserves_optimizer_state(tmp_path):
    """After a few optimizer steps, save/load should leave model+optimizer
    state byte-identical (within float comparison tolerance)."""
    from model.utils import save_checkpoint, load_checkpoint
    torch.manual_seed(0)
    model = _make_tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(2, 4)).pow(2).sum().backward()
        opt.step()
    ckpt = tmp_path / "round.pt"
    save_checkpoint(model, opt, iteration=99, out=str(ckpt))

    model2 = _make_tiny_model()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    it = load_checkpoint(str(ckpt), model2, opt2)
    assert it == 99
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        torch.testing.assert_close(p1.data, p2.data)
    # Optimizer state mtrx should also match
    s1 = opt.state_dict()["state"]
    s2 = opt2.state_dict()["state"]
    for k in s1:
        for sk, v in s1[k].items():
            if torch.is_tensor(v):
                torch.testing.assert_close(v, s2[k][sk])
