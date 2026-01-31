import torch
import triton
import triton.language as tl


@triton.jit
def _segment_reduce_weighted_kernel(
    # Input pointers
    expert_output_ptr,      # (num_sorted, d_model) - expert outputs sorted by target token
    sorted_weights_ptr,     # (num_sorted,) - weights sorted by target token
    segment_offsets_ptr,    # (num_tokens + 1,) - CSR-style offsets for each token's segment
    # Output pointer
    y_flat_ptr,             # (num_tokens, d_model) - output
    # Dimensions
    num_tokens,
    d_model,
    stride_expert_row,
    stride_y_row,
    # Block sizes
    BLOCK_D: tl.constexpr,
    MAX_TOPK: tl.constexpr,  # max number of experts per token (usually 1-4)
):
    """
    Segment reduction kernel: sum contributions for each output token.
    
    Unlike atomic scatter-add, this kernel:
    1. Processes one OUTPUT token per program (not one input position)
    2. Reads the segment of inputs that map to this output
    3. Sums them up (no atomics needed!)
    
    This is MUCH faster because:
    - Reads are coalesced (segment is contiguous after sorting)
    - Writes are coalesced (one output row per program)  
    - No atomic contention
    
    Grid: (num_tokens, ceil(d_model / BLOCK_D))
    """
    token_id = tl.program_id(0)    # which output token
    d_block = tl.program_id(1)     # which d_model block
    
    if token_id >= num_tokens:
        return
    
    # Load segment bounds for this token
    seg_start = tl.load(segment_offsets_ptr + token_id)
    seg_end = tl.load(segment_offsets_ptr + token_id + 1)
    
    # d_model offsets for this block
    d_offs = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < d_model
    
    # Accumulate contributions from this segment (in fp32 for precision)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    # Unrolled loop over segment (MAX_TOPK is small, typically 1-4)
    for k in range(MAX_TOPK):
        pos = seg_start + k
        if pos < seg_end:
            # Load weight and expert output
            weight = tl.load(sorted_weights_ptr + pos)
            expert_out = tl.load(
                expert_output_ptr + pos * stride_expert_row + d_offs,
                mask=d_mask, other=0.0
            )
            acc += expert_out.to(tl.float32) * weight.to(tl.float32)
    
    # Store result (convert back to input dtype - bf16/fp16/fp32)
    # Note: the dtype is inferred from the output pointer type
    tl.store(y_flat_ptr + token_id * stride_y_row + d_offs, acc, mask=d_mask)


def segment_reduce_weighted(
    expert_output: torch.Tensor,    # (num_sorted, d_model) - outputs sorted by target token
    sorted_weights: torch.Tensor,   # (num_sorted,) - weights sorted by target token
    sorted_token_idx: torch.Tensor, # (num_sorted,) - target indices (sorted)
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """
    Efficient scatter-add using segment reduction (NO atomics).
    
    Prerequisite: The inputs must be sorted by target token index!
    This allows us to process each output token's contributions as a contiguous segment.
    
    Args:
        expert_output: Expert outputs sorted by target token, shape (num_sorted, d_model)
        sorted_weights: Weights sorted by target token, shape (num_sorted,)
        sorted_token_idx: Target token indices (must be sorted!), shape (num_sorted,)
        num_tokens: Number of output tokens
        top_k: Maximum experts per token (for loop unrolling)
    
    Returns:
        y_flat: Accumulated output, shape (num_tokens, d_model)
    """
    num_sorted, d_model = expert_output.shape
    device = expert_output.device
    dtype = expert_output.dtype
    
    # Compute segment offsets (CSR format): where each token's segment starts/ends
    # Since sorted_token_idx is sorted, we can use searchsorted
    token_ids = torch.arange(num_tokens + 1, device=device, dtype=torch.int64)
    segment_offsets = torch.searchsorted(sorted_token_idx.contiguous(), token_ids)
    
    # Allocate output (use fp32 for kernel accumulation, convert at end)
    # This matches the kernel's fp32 accumulator and avoids precision loss
    y_flat_fp32 = torch.empty(num_tokens, d_model, device=device, dtype=torch.float32)
    
    # Choose block size
    BLOCK_D = min(triton.next_power_of_2(d_model), 256)
    MAX_TOPK = triton.next_power_of_2(max(top_k, 2))  # at least 2 for unrolling
    
    # Grid: one program per (token, d_block)
    grid = (num_tokens, triton.cdiv(d_model, BLOCK_D))
    
    _segment_reduce_weighted_kernel[grid](
        expert_output, sorted_weights, segment_offsets,
        y_flat_fp32,
        num_tokens, d_model,
        expert_output.stride(0), y_flat_fp32.stride(0),
        BLOCK_D=BLOCK_D,
        MAX_TOPK=MAX_TOPK,
    )
    
    # Convert back to original dtype
    return y_flat_fp32.to(dtype)


def fused_scatter_add_weighted(
    expert_output: torch.Tensor,    # (num_sorted, d_model) - raw expert outputs
    sorted_token_idx: torch.Tensor, # (num_sorted,) - target token indices (NOT sorted by target)
    sorted_weights: torch.Tensor,   # (num_sorted,) - weights
    num_tokens: int,                # number of original tokens
    top_k: int = 2,                 # experts per token (for optimization hints)
) -> torch.Tensor:
    """
    Optimized scatter-add with weight multiplication using segment reduction.
    
    This function re-sorts the data by target token to enable efficient 
    segment reduction (no atomic operations needed).
    
    Computes: y_flat[idx, :] = sum(expert_output[i, :] * weight[i]) for all i where idx[i] == token
    
    Performance characteristics:
    - Sort by target: O(N log N) but highly optimized on GPU
    - Segment reduction: O(N * d_model / parallelism), fully coalesced, no atomics
    - Much faster than atomic scatter-add for typical MoE configurations
    
    Args:
        expert_output: Expert outputs before weighting, shape (num_sorted, d_model)
        sorted_token_idx: Target token indices, shape (num_sorted,)
        sorted_weights: Routing weights, shape (num_sorted,)
        num_tokens: Number of original tokens (for output shape)
        top_k: Number of experts per token (optimization hint)
    
    Returns:
        y_flat: Accumulated output, shape (num_tokens, d_model)
    """
    # Re-sort by target token index to enable segment reduction
    sort_perm = torch.argsort(sorted_token_idx)
    expert_output_sorted = expert_output[sort_perm]
    weights_sorted = sorted_weights[sort_perm]
    target_idx_sorted = sorted_token_idx[sort_perm]
    
    # Use segment reduction (no atomics!)
    return segment_reduce_weighted(
        expert_output_sorted,
        weights_sorted,
        target_idx_sorted,
        num_tokens,
        top_k,
    )
