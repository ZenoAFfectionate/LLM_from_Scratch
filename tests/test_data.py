import math
from collections import Counter

import numpy as np
import pytest

from .adapters import run_get_batch


def test_get_batch():
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 32
    device = "cpu"

    # Sanity check to make sure that the random samples are indeed somewhat random.
    starting_indices = Counter()
    num_iters = 1000
    for _ in range(num_iters):
        x, y = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # Make sure the shape is correct
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)

        # Make sure the y's are always offset by 1
        np.testing.assert_allclose((x + 1).detach().numpy(), y.detach().numpy())

        starting_indices.update(x[:, 0].tolist())

    # Make sure we never sample an invalid start index
    num_possible_starting_indices = len(dataset) - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0
    # Expected # of times that we see each starting index
    expected_count = (num_iters * batch_size) / num_possible_starting_indices
    standard_deviation = math.sqrt(
        (num_iters * batch_size) * (1 / num_possible_starting_indices) * (1 - (1 / num_possible_starting_indices))
    )
    # Range for expected outcomes (mu +/- 5sigma). For a given index,
    # this should happen 99.99994% of the time of the time.
    # So, in the case where we have 93 possible start indices,
    # the entire test should pass with 99.9944202% of the time
    occurrences_lower_bound = expected_count - 5 * standard_deviation
    occurrences_upper_bound = expected_count + 5 * standard_deviation

    for starting_index, count in starting_indices.items():
        if count < occurrences_lower_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at least {occurrences_lower_bound}"
            )
        if count > occurrences_upper_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at most {occurrences_upper_bound}"
            )

    with pytest.raises((RuntimeError, AssertionError)) as excinfo:
        # We're assuming that cuda:99 is an invalid device ordinal.
        # Just adding this here to make sure that the device flag is
        # being handled.
        run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device="cuda:99",
        )
        assert "CUDA error" in str(excinfo.value) or "Torch not compiled with CUDA enabled" in str(excinfo.value)


# =============================================================================
# Direct tests for model.utils.data_loading (no adapter shim).
# =============================================================================

def test_data_loading_shapes_and_dtype():
    from model.utils import data_loading
    rng = np.random.RandomState(0)
    data = rng.randint(0, 100, size=200).astype(np.int64)
    bsz, ctx = 4, 16
    inputs, targets = data_loading(data, bsz, ctx, device="cpu")
    assert inputs.shape == (bsz, ctx) and targets.shape == (bsz, ctx)
    assert inputs.dtype == np.dtype("int64") or inputs.dtype.is_floating_point is False
    # torch dtypes
    import torch
    assert inputs.dtype == torch.long and targets.dtype == torch.long


def test_data_loading_target_is_input_plus_one():
    """When the source is the identity sequence 0..N-1, targets must equal inputs+1."""
    import torch
    from model.utils import data_loading
    data = np.arange(200, dtype=np.int64)
    inputs, targets = data_loading(data, batch_size=8, context_length=16, device="cpu")
    assert torch.equal(targets, inputs + 1)


def test_data_loading_indices_within_bounds():
    """Even at the boundary of the data, targets must stay within the array."""
    from model.utils import data_loading
    data = np.arange(50, dtype=np.int64)
    inputs, targets = data_loading(data, batch_size=8, context_length=10, device="cpu")
    assert int(targets.max()) <= 49
    assert int(inputs.min()) >= 0
