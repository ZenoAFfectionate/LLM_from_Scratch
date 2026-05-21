import math

import numpy
import pytest
import torch

from .adapters import get_adamw_cls, run_get_lr_cosine_schedule

# Direct imports for the dedicated optimizer-class tests added below.
# These bypass the adapter so they remain useful even when the adapter
# import surface drifts from the implementation.
from model.optimizer.SGD import SGD
from model.optimizer.AdamW import AdamW
from model.optimizer.Muon import newtonschulz5_orthogonalization, Muon


def _optimize(opt_class) -> torch.Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # Use 1000 optimization steps for testing
    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


def test_adamw(numpy_snapshot):
    """
    Our reference implementation yields slightly different results than the
    PyTorch AdamW, since there are a couple different ways that you can apply
    weight decay that are equivalent in principle, but differ in practice due to
    floating point behavior. So, we test that the provided implementation matches
    _either_ our reference implementation's expected results or those from the PyTorch AdamW.
    """
    # expected_weights = torch.load(FIXTURES_PATH / "adamw_expected_params.pt")
    pytorch_weights = _optimize(torch.optim.AdamW)
    actual_weights = _optimize(get_adamw_cls())

    # Might need to exit early if the weights match pytorch, since that should also be valid
    matches_pytorch = torch.allclose(actual_weights, pytorch_weights, atol=1e-4)
    if matches_pytorch:
        return

    numpy_snapshot.assert_match(
        actual_weights,
        atol=1e-4,
    )


def test_get_lr_cosine_schedule():
    max_learning_rate = 1
    min_learning_rate = 1 * 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    expected_lrs = [
        0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]
    actual_lrs = [
        run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for it in range(25)
    ]
    numpy.testing.assert_allclose(numpy.array(actual_lrs), numpy.array(expected_lrs))


# =============================================================================
# Additional optimizer-class tests (SGD / AdamW / Muon)
# These import the implementations directly to test their public behaviour
# independently of the tests/adapters.py shim.
# =============================================================================


# ----------------------------------------------------------------------- SGD

def test_sgd_invalid_lr_rejected():
    p = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError):
        SGD([p], lr=-0.1)


def test_sgd_step_uses_decayed_lr():
    """The SGD here uses lr_eff = lr / sqrt(t+1). Verify update is exact."""
    w = torch.nn.Parameter(torch.tensor([10.0]))
    opt = SGD([w], lr=1.0)
    expected = 10.0
    for t in range(5):
        opt.zero_grad()
        (w ** 2).backward()                          # grad = 2w
        grad = 2 * expected
        lr_eff = 1.0 / math.sqrt(t + 1)
        expected = expected - lr_eff * grad
        opt.step()
        assert w.item() == pytest.approx(expected, rel=1e-6)


def test_sgd_converges_on_quadratic():
    w = torch.nn.Parameter(torch.tensor([5.0]))
    opt = SGD([w], lr=0.5)
    for _ in range(80):
        opt.zero_grad()
        (w ** 2).backward()
        opt.step()
    assert abs(w.item()) < 0.5


# --------------------------------------------------------------------- AdamW

def test_adamw_arg_validation():
    p = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError):
        AdamW([p], lr=-1)
    with pytest.raises(ValueError):
        AdamW([p], eps=-1)
    with pytest.raises(ValueError):
        AdamW([p], betas=(1.5, 0.9))
    with pytest.raises(ValueError):
        AdamW([p], betas=(0.9, -0.1))
    with pytest.raises(ValueError):
        AdamW([p], weight_decay=-1)


def test_adamw_first_step_sign_correct():
    """Positive gradient + positive lr must reduce the parameter on step 1."""
    w = torch.nn.Parameter(torch.tensor([5.0]))
    opt = AdamW([w], lr=0.1, weight_decay=0.0)
    opt.zero_grad()
    (w ** 2).backward()
    initial = w.item()
    opt.step()
    assert w.item() < initial


def test_adamw_state_is_lazy_and_fp32():
    """State is allocated only after the first step, and m/v are always fp32."""
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    opt = AdamW([w], lr=1e-3)
    assert len(opt.state[w]) == 0
    (w ** 2).sum().backward()
    opt.step()
    st = opt.state[w]
    assert {"step", "m", "v"} <= set(st.keys())
    assert st["m"].dtype == torch.float32
    assert st["v"].dtype == torch.float32
    assert st["step"] == 1


def test_adamw_matches_torch_within_tolerance():
    """Custom AdamW vs torch.optim.AdamW from identical init should agree closely."""
    torch.manual_seed(0)
    init = torch.randn(20)

    def run(opt_cls):
        w = torch.nn.Parameter(init.clone())
        opt = opt_cls([w], lr=1e-3)
        for _ in range(30):
            opt.zero_grad()
            (w ** 2).sum().backward()
            opt.step()
        return w.detach().clone()

    ours = run(AdamW)
    theirs = run(torch.optim.AdamW)
    torch.testing.assert_close(ours, theirs, atol=1e-4, rtol=1e-3)


def test_adamw_converges_on_quadratic():
    w = torch.nn.Parameter(torch.tensor([5.0]))
    opt = AdamW([w], lr=0.3, weight_decay=0.0)
    for _ in range(120):
        opt.zero_grad()
        (w ** 2).backward()
        opt.step()
    assert abs(w.item()) < 0.5


# ---------------------------------------------------------------------- Muon

def test_newtonschulz_orthogonalises_random_matrix():
    """X X^T should be close to identity after NS iterations."""
    if not torch.cuda.is_available():
        pytest.skip("Newton-Schulz uses bf16 internally; needs CUDA")
    torch.manual_seed(0)
    G = torch.randn(32, 32, device="cuda")
    X = newtonschulz5_orthogonalization(G, steps=8).float()
    XXt = X @ X.transpose(-1, -2)
    eye = torch.eye(32, device="cuda")
    off = (XXt - eye).abs()
    # Quintic NS with overshoot coefficients pushes singular values past 1; tolerate up to 0.4.
    assert off.max() < 0.4, f"X X^T not close to I: max off = {off.max():.3f}"


def test_newtonschulz_handles_rectangular():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    for shape in [(8, 32), (64, 16)]:
        G = torch.randn(*shape, device="cuda")
        X = newtonschulz5_orthogonalization(G, steps=6)
        assert X.shape == G.shape


def test_muon_optimizer_step_reduces_loss():
    """Full Muon update on a 2D parameter should decrease a quadratic loss."""
    if not torch.cuda.is_available():
        pytest.skip("Muon kernel uses bf16 NS on CUDA")
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(8, 8, device="cuda"))
    opt = Muon([W], lr=0.05, momentum=0.9, ns_steps=5)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = (W ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8, (
        f"Muon did not reduce loss enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
    )


# --------------------------------------------------- LR schedule edge cases

def test_cos_lr_warmup_phase_is_linear():
    """Direct invocation of cos_learning_rate_schedule_with_warmup."""
    from model.utils import cos_learning_rate_schedule_with_warmup
    for t in range(0, 100, 7):
        lr = cos_learning_rate_schedule_with_warmup(t, 1.0, 0.1, 100, 1000)
        assert lr == pytest.approx(t / 100.0)


def test_cos_lr_post_anneal_is_min():
    from model.utils import cos_learning_rate_schedule_with_warmup
    lr = cos_learning_rate_schedule_with_warmup(10_000, 1.0, 0.1, 100, 1000)
    assert lr == 0.1


def test_cos_lr_monotonic_decreasing_in_anneal():
    from model.utils import cos_learning_rate_schedule_with_warmup
    lrs = [
        cos_learning_rate_schedule_with_warmup(t, 1.0, 0.0, 0, 100)
        for t in range(100)
    ]
    diffs = numpy.diff(lrs)
    assert (diffs <= 1e-9).all(), "LR must be non-increasing during cosine anneal"
