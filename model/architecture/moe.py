import math
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

from model.architecture.mlp import MLP
from model.architecture.kernels import fused_scatter_add_weighted


class Gate(nn.Module):
    """ PyTorch implementation of MoE Gate mechanism with Auxiliary-Loss-Free Load Balancing """

    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        bias_update_speed: float = 0.01,
        aux_seq_loss_alpha: float = 0.01,
        device=None,
        dtype=None
    ):
        super().__init__()
        # parameters for MoE gating
        self.hidden_size = hidden_size
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        # parameters for load balancing
        self.seq_alpha = aux_seq_loss_alpha
        self.bias_update_speed = bias_update_speed

        # [OPT] expert_bias is kept in fp32 for precision during updates
        # We cache a converted version to avoid per-forward dtype conversion
        self.register_buffer('expert_bias', torch.zeros(
            self.n_routed_experts, device=device, dtype=torch.float32))
        self.register_buffer('expert_load', torch.zeros(
            self.n_routed_experts, device=device, dtype=torch.long))
        # [OPT] Cache for converted bias - lazily populated on first forward
        self._cached_bias_dtype = None
        self._cached_bias = None

        # initialize gating weights for affinity score calculation
        self.weight = nn.Parameter(torch.empty(
            (self.n_routed_experts, self.hidden_size), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Invalidate cache when parameters are reset
        self._cached_bias_dtype = None
        self._cached_bias = None

    def _get_bias_in_dtype(self, target_dtype: torch.dtype) -> torch.Tensor:
        """Get expert_bias in the target dtype, using cache to avoid repeated conversions.

        [OPT] During training, bias changes after each step via update_bias().
        During inference, bias is constant. The cache avoids repeated .to() calls.
        """
        if self._cached_bias_dtype != target_dtype or self._cached_bias is None:
            self._cached_bias = self.expert_bias.to(target_dtype)
            self._cached_bias_dtype = target_dtype
        return self._cached_bias

    def forward(self, x):
        """ Forward pass with auxiliary-loss-free load balancing """
        bsz, seq_len, _ = x.shape

        x_flat = x.view(-1, x.shape[-1])
        # calculate affinity scores for each expert
        logits = x_flat @ self.weight.t()
        scores = torch.sigmoid(logits)

        # [OPT] Use cached bias conversion to avoid repeated .to() calls
        # The cache is invalidated when update_bias() is called
        if self.expert_bias.dtype == logits.dtype:
            biased_logits = logits + self.expert_bias.unsqueeze(0)
        else:
            biased_logits = logits + \
                self._get_bias_in_dtype(logits.dtype).unsqueeze(0)

        # select top-k experts based on biased logits and their unbiased weights
        _, topk_indices = torch.topk(
            biased_logits, k=self.top_k, dim=-1, sorted=False)
        topk_weights = torch.gather(scores, dim=-1, index=topk_indices)
        if self.top_k > 1:  # re-normalize the weights of the selected
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-10
            topk_weights = topk_weights / denominator

        # compute expert_load ONCE here, reuse to avoid duplicate
        # [OPT] Use one_hot + sum instead of bincount - faster for small n_routed_experts
        expert_token_counts = None
        if self.training:
            # one_hot creates (N, n_experts) then sum along dim=0 gives counts per expert
            expert_token_counts = F.one_hot(
                topk_indices.flatten(),
                num_classes=self.n_routed_experts
            ).sum(dim=0)
            self.expert_load = expert_token_counts

        # calculate sequence-wise auxiliary loss (if alpha > 0)
        aux_seq_loss = None
        if self.seq_alpha > 0 and self.training:
            aux_seq_loss = self._compute_sequence_balance_loss(
                scores, topk_indices, bsz, seq_len
            )

        # return expert_token_counts to avoid recomputation in MOE.forward()
        return topk_indices, topk_weights, aux_seq_loss, expert_token_counts

    def _compute_sequence_balance_loss(self, scores, topk_idx, bsz, seq_len):
        """
        Compute sequence-wise balance loss as described in DeepSeek-V3.
        Fully vectorized implementation - no Python loops, no GPU-CPU sync

        The loss encourages balanced expert load within each sequence:
        L_Bal = α * Σ(f_i * P_i)

        where:
        - f_i: fraction of tokens in sequence where expert i is in top-K (eq. 18)
        - P_i: average of normalized routing probabilities for expert i (eq. 19-20)
        - α: small hyperparameter weight (seq_alpha)
        """
        scores_reshaped = scores.view(
            bsz, seq_len, self.n_routed_experts)  # (bsz, seq_len, n_experts)
        topk_idx_reshaped = topk_idx.view(
            bsz, seq_len, self.top_k)         # (bsz, seq_len, top_k)

        # Use F.one_hot - output is already float when scores is float
        expert_mask = F.one_hot(
            topk_idx_reshaped.view(bsz, -1),  # (bsz, seq_len*top_k)
            num_classes=self.n_routed_experts
        )  # (bsz, seq_len*top_k, n_experts), dtype=int64

        # Compute f_i directly with int64 sum, then convert to float at the end
        expert_counts = expert_mask.sum(dim=1)  # (bsz, n_experts)
        f_i = expert_counts.to(scores.dtype) / \
            (self.top_k * seq_len)  # (bsz, n_experts)

        # vectorized P_i calculation: (bsz, n_experts):
        score_sums = scores_reshaped.sum(
            dim=2, keepdim=True)       # (bsz, seq_len, 1)
        normalized_scores = scores_reshaped / \
            (score_sums + 1e-10)  # (bsz, seq_len, n_experts)
        P_i = normalized_scores.mean(dim=1)  # (bsz, n_experts)

        seq_losses = (f_i * P_i).sum(dim=1)  # (bsz,)
        aux_seq_loss = self.seq_alpha * seq_losses.mean()

        return aux_seq_loss

    def update_bias(self, total_tokens):
        """
        Update expert bias vectorized based on load balance.
        Should be called at the end of each training step.

        Args:
            total_tokens: Total number of tokens processed in the batch
        """
        if not self.training:
            return
        # calculate expected load per expert (uniform distribution)
        expected_load = (total_tokens * self.top_k) / self.n_routed_experts
        # vectorized bias update: compute difference and update all biases at once
        load_diff = self.expert_load.float() - expected_load
        self.expert_bias -= torch.sign(load_diff) * self.bias_update_speed

        # [OPT] Invalidate cached bias since we just updated it
        self._cached_bias_dtype = None
        self._cached_bias = None


class MOE(nn.Module):
    """
    Mixture of Experts Feed-Forward Network with Auxiliary-Loss-Free Load Balancing
    In this implementation, we optimize the training path using a sort-based approach
    to achieve better memory access patterns and scalability with number of experts. 
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_shared_experts: int = 0,
        bias_update_speed: float = 0.01,
        aux_seq_loss_alpha: float = 0.01,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        # initialize experts and gate
        self.experts = nn.ModuleList([
            MLP(d_model, d_ff, device=device, dtype=dtype)
            for _ in range(n_routed_experts)
        ])
        self.gate = Gate(
            hidden_size=d_model,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            aux_seq_loss_alpha=aux_seq_loss_alpha,
            bias_update_speed=bias_update_speed,
            device=device,
            dtype=dtype
        )
        # Initialize shared expert (single MLP)
        if n_shared_experts > 0:
            self.shared_expert = MLP(d_model, d_ff, device=device, dtype=dtype)

    # @torch._dynamo.disable()
    def forward(self, x):
        """
        MoE forward pass with torch.compile disabled.

        Reason: MoE routing is inherently dynamic - the number of tokens per expert
        varies at runtime. The .tolist() call required by torch.split() causes
        unavoidable graph breaks. Running in eager mode avoids recompilation overhead.
        """
        identity = x
        batch_size, seq_len, _ = x.shape

        # choose expert using gate mechanism (auxiliary sequence-wise loss)
        topk_idx, topk_weight, aux_seq_loss, expert_token_counts = self.gate(x)
        # reshape for token-level processing
        x_flat = x.view(-1, x.shape[-1])   # (batch_size * seq_len, d_model)
        flat_topk_idx = topk_idx.view(-1)  # (batch_size * seq_len * top_k,)

        if self.training:
            n_total_tokens = batch_size * seq_len
            flat_topk_weight = topk_weight.view(-1)

            # create token indices for each expert selection to represents
            # which original token each selection belongs to (batch*seq*top_k,)
            token_indices = torch.arange(
                n_total_tokens, device=x_flat.device
            ).repeat_interleave(self.num_experts_per_tok)

            # sort by expert index to create contiguous chunks: O(N log N)
            # this will groups all tokens for Expert 0, then Expert 1, etc.
            sorted_expert_idx = torch.argsort(flat_topk_idx)
            sorted_token_idx = token_indices[sorted_expert_idx]
            sorted_weight = flat_topk_weight[sorted_expert_idx]

            # permute tokens to match expert group for contiguous memory access
            sorted_tokens = x_flat[sorted_token_idx]

            # Reuse expert_token_counts from Gate - single sync point here
            # Convert counts to list ONCE for torch.split and loop bounds
            expert_counts_list = expert_token_counts.tolist()

            # Split sorted_tokens into chunks by expert (no additional sync needed)
            # torch.split returns a tuple of tensors, one per expert
            expert_inputs = torch.split(sorted_tokens, expert_counts_list)

            # Process each expert and collect outputs
            expert_outputs = [
                self.experts[expert_id](expert_inputs[expert_id])
                for expert_id in range(self.n_routed_experts)
            ]

            # Concatenate all outputs back together (same order as sorted_tokens)
            y_sorted = torch.cat(expert_outputs, dim=0)

            # Fused Triton kernel: weight multiplication + scatter-add using segment reduction
            # This is MUCH faster than atomic scatter-add because:
            #   1. Re-sorts by target token to enable contiguous segment access
            #   2. Uses segment reduction (no atomics!) - each output token sums its segment
            #   3. Fully coalesced reads and writes
            y_flat = fused_scatter_add_weighted(
                y_sorted,           # raw expert outputs
                sorted_token_idx,   # target token indices
                sorted_weight,      # routing weights
                n_total_tokens,     # number of original tokens
                self.num_experts_per_tok  # top_k for segment loop unrolling
            )

            y = y_flat.view(batch_size, seq_len, -1)
        else:
            # use existing optimized inference path
            y = self.moe_infer(x_flat, flat_topk_idx, topk_weight.view(-1, 1))
            y = y.view(batch_size, seq_len, -1)

        # Apply shared expert and add to output
        if self.n_shared_experts > 0:
            y = y + self.shared_expert(identity)

        self.aux_loss = aux_seq_loss
        self._total_tokens = batch_size * seq_len

        return y

    def update_expert_bias(self):
        """
        Update expert bias based on load balance.
        Should be called at the end of each training step.
        """
        if hasattr(self, '_total_tokens'):
            self.gate.update_bias(self._total_tokens)

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """Optimized inference logic for Mixture-of-Experts

        [OPT] Optimizations applied:
        1. Pre-convert weights dtype once instead of per-expert
        2. Use one_hot + sum instead of bincount
        3. Use torch.split to avoid cumsum + tolist overhead
        4. Use expand instead of repeat for memory efficiency
        """
        expert_cache = torch.zeros_like(x)

        # sort indices to group tokens by expert
        idxs = flat_expert_indices.argsort()

        # Use one_hot + sum instead of bincount: faster for small n_experts
        counts = F.one_hot(
            flat_expert_indices,
            num_classes=self.n_routed_experts
        ).sum(dim=0)

        # Single sync point - get counts as list for torch.split
        counts_list = counts.tolist()

        token_idxs = idxs // self.num_experts_per_tok

        # [OPT] Pre-convert weights to match x dtype once
        if flat_expert_weights.dtype != x.dtype:
            flat_expert_weights = flat_expert_weights.to(x.dtype)

        # [OPT] Pre-compute d_model for index expansion
        d_model = x.shape[-1]

        # Split sorted indices into per-expert chunks
        token_idx_chunks = torch.split(token_idxs, counts_list)
        weight_idx_chunks = torch.split(idxs, counts_list)

        # loop through the batches of tokens for each expert
        # [OPT] Remove data-dependent `if count == 0` check to avoid graph breaks
        # All operations (indexing, expert forward, scatter_add_) handle empty tensors correctly
        for i in range(self.n_routed_experts):
            expert = self.experts[i]
            exp_token_idx = token_idx_chunks[i]

            # get the batch of tokens for this expert (empty tensor if no tokens)
            expert_tokens = x[exp_token_idx]

            # process the batch and weight the output (works with empty tensors)
            expert_out = expert(expert_tokens)

            # [OPT] weights already in correct dtype, no conversion needed
            weights = flat_expert_weights[weight_idx_chunks[i]]
            expert_out = expert_out * weights

            # [OPT] Use expand instead of repeat for memory efficiency (no copy)
            # scatter_add_ with empty indices is a no-op (correct behavior)
            scatter_idx = exp_token_idx.unsqueeze(1).expand(-1, d_model)
            expert_cache.scatter_add_(0, scatter_idx, expert_out)

        return expert_cache
