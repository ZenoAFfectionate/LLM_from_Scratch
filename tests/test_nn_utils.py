import numpy
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from .adapters import run_cross_entropy, run_gradient_clipping, run_softmax


def test_softmax_matches_pytorch():
    x = torch.tensor(
        [
            [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
            [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
            [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
        ]
    )
    expected = F.softmax(x, dim=-1)
    numpy.testing.assert_allclose(run_softmax(x, dim=-1).detach().numpy(), expected.detach().numpy(), atol=1e-6)
    # Test that softmax handles numerical overflow issues
    numpy.testing.assert_allclose(
        run_softmax(x + 100, dim=-1).detach().numpy(),
        expected.detach().numpy(),
        atol=1e-6,
    )


def test_cross_entropy():
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    numpy.testing.assert_allclose(
        run_cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1)).detach().numpy(),
        expected.detach().numpy(),
        atol=1e-4,
    )

    # Test that cross-entropy handles numerical overflow issues
    large_inputs = 1000.0 * inputs
    large_expected_cross_entropy = F.cross_entropy(large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1))
    numpy.testing.assert_allclose(
        run_cross_entropy(large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1)).detach().numpy(),
        large_expected_cross_entropy.detach().numpy(),
        atol=1e-4,
    )


def test_gradient_clipping():
    tensors = [torch.randn((5, 5)) for _ in range(6)]
    max_norm = 1e-2

    t1 = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    # Test freezing one parameter.
    t1[-1].requires_grad_(False)

    loss = torch.cat(t1).sum()
    loss.backward()
    clip_grad_norm_(t1, max_norm)
    t1_grads = [torch.clone(t.grad) for t in t1 if t.grad is not None]

    t1_c = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    t1_c[-1].requires_grad_(False)
    loss_c = torch.cat(t1_c).sum()
    loss_c.backward()
    run_gradient_clipping(t1_c, max_norm)
    t1_c_grads = [torch.clone(t.grad) for t in t1_c if t.grad is not None]

    assert len(t1_grads) == len(t1_c_grads)

    for t1_grad, t1_c_grad in zip(t1_grads, t1_c_grads):
        numpy.testing.assert_allclose(
            t1_grad.detach().numpy(),
            t1_c_grad.detach().numpy(),
            atol=1e-6,
        )


# =============================================================================
# Additional direct tests for model/utils.py NN helpers
# (Embedding, RMSNorm, ResScale, cross_entropy stability, softmax sum).
# =============================================================================

def test_softmax_output_sums_to_one():
    from model.attention.utils import softmax
    x = torch.randn(3, 5, 7)
    sums = softmax(x, dim=-1).sum(dim=-1)
    numpy.testing.assert_allclose(sums.numpy(), numpy.ones((3, 5)), atol=1e-6)


def test_cross_entropy_minimised_by_correct_one_hot():
    """Cranking the logit at the target index to a very large value drives CE → 0."""
    from model.utils import cross_entropy
    logits = torch.zeros(3, 5)
    targets = torch.tensor([1, 4, 0])
    logits[range(3), targets] = 50.0
    assert cross_entropy(logits, targets).item() < 1e-10


def test_cross_entropy_overflow_safe():
    """Shifting all logits by a large constant must not change CE (LSE invariance)."""
    from model.utils import cross_entropy
    torch.manual_seed(0)
    logits = torch.randn(5, 9)
    targets = torch.randint(0, 9, (5,))
    base = cross_entropy(logits, targets)
    shifted = cross_entropy(logits + 1000.0, targets)
    numpy.testing.assert_allclose(base.item(), shifted.item(), atol=1e-4)


# ----------------------------------------------------------------- Embedding

def test_embedding_matches_indexing():
    from model.utils import Embedding
    layer = Embedding(50, 8)
    ids = torch.tensor([[1, 2, 3], [0, 49, 7]])
    out = layer(ids)
    assert out.shape == (2, 3, 8)
    assert torch.equal(out, layer.weight[ids])


def test_embedding_int32_ids_accepted():
    from model.utils import Embedding
    layer = Embedding(10, 4)
    ids = torch.tensor([1, 2], dtype=torch.int32)
    assert layer(ids).shape == (2, 4)


def test_embedding_grad_only_on_used_rows():
    from model.utils import Embedding
    layer = Embedding(8, 4)
    ids = torch.tensor([0, 3, 7])
    layer(ids).pow(2).sum().backward()
    nz = (layer.weight.grad.abs().sum(-1) > 0).tolist()
    assert nz == [True, False, False, True, False, False, False, True]


# ----------------------------------------------------------------- RMSNorm

def test_rmsnorm_matches_reference():
    from model.utils import RMSNorm
    d, eps = 16, 1e-5
    layer = RMSNorm(d, eps=eps)
    layer.weight.data.copy_(torch.linspace(0.5, 2.0, d))
    x = torch.randn(3, 5, d)
    actual = layer(x)
    ref_rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    expected = (x.float() * ref_rms * layer.weight).to(x.dtype)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_rmsnorm_fused_residual_path():
    """RMSNorm(x, residual=r) returns (norm(x+r), x+r)."""
    from model.utils import RMSNorm
    d = 8
    layer = RMSNorm(d)
    x = torch.randn(2, 4, d)
    res = torch.randn(2, 4, d)
    normed, new_res = layer(x, residual=res)
    torch.testing.assert_close(new_res, (x + res).to(new_res.dtype))
    expected_norm = layer(x + res)
    torch.testing.assert_close(normed, expected_norm, atol=1e-6, rtol=1e-5)


def test_rmsnorm_preserves_input_dtype():
    from model.utils import RMSNorm
    layer = RMSNorm(8)
    for dt in [torch.float32, torch.bfloat16]:
        y = layer(torch.randn(2, 4, 8, dtype=dt))
        assert y.dtype == dt


# ----------------------------------------------------------------- ResScale

def test_resscale_is_identity_at_init():
    from model.utils import ResScale
    layer = ResScale(d_model=12)
    x = torch.randn(3, 5, 12)
    torch.testing.assert_close(layer(x), x)


def test_resscale_applies_affine_after_update():
    from model.utils import ResScale
    layer = ResScale(d_model=4)
    layer.alpha.data.fill_(2.0)
    layer.beta.data.fill_(0.5)
    x = torch.ones(1, 1, 4)
    torch.testing.assert_close(layer(x), torch.full((1, 1, 4), 2.5))
