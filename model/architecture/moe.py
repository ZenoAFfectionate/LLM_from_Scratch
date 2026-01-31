import math
import torch
import triton

import torch.nn as nn
import triton.language as tl
import torch.nn.functional as F

from model.architecture.mlp import MLP


class FusedSharedExperts(nn.Module):
    """
    Fused implementation of multiple shared experts. Instead of running N separate experts
    sequentially (launch N kernel), we fuse them into one larger expert and split outputs.d

    Example: 2 experts × (d_model → d_ff → d_model) becomes:
             1 expert × (d_model → 2*d_ff → d_model)

    Benefits:
    - Reduces CUDA kernel launch overhead by N×
    - Better memory coalescing and cache utilization
    - Single large matmul is more efficient than N small matmuls
    """
    def __init__(self, d_model: int, d_ff: int, n_shared_experts: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_shared_experts = n_shared_experts

        # fused linear layers with expanded dimensions
        fused_d_ff = d_ff * n_shared_experts
        self.w1 = nn.Linear(d_model, fused_d_ff, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, fused_d_ff, device=device, dtype=dtype)
        self.w2 = nn.Linear(fused_d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through fused shared experts. The computation is equivalent 
        to summing N individual experts but done in a single pass for efficiency.
        """
        # Single forward pass through fused dimensions
        w1_out = self.w1(x)  # (batch, seq, d_ff * n_experts)
        w3_out = self.w3(x)  # (batch, seq, d_ff * n_experts)
        activated = F.silu(w1_out) * w3_out
        return self.w2(activated)  # (batch, seq, d_model)


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
   
        self.register_buffer('expert_bias', torch.zeros(self.n_routed_experts, device=device, dtype=torch.float32))
        self.register_buffer('expert_load', torch.zeros(self.n_routed_experts, device=device, dtype=torch.long))
        # initialize gating weights for affinity score calculation
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        """ Forward pass with auxiliary-loss-free load balancing """
        bsz, seq_len, _ = x.shape

        # calculate affinity scores for each expert
        x_flat = x.view(-1, x.shape[-1])
        # use @ operator for compile optimization
        logits = x_flat @ self.weight.t()
        scores = torch.sigmoid(logits)

        # add bias term to affinity scores for top-k routing (only for routing)
        biased_logits = logits + self.expert_bias.to(logits.dtype).unsqueeze(0)

        # select top-k experts based on biased logits and their unbiased weights
        _, topk_indices = torch.topk(biased_logits, k=self.top_k, dim=-1, sorted=False)
        topk_weights = torch.gather(scores, dim=-1, index=topk_indices)
        # re-normalize the weights of the selected experts
        if self.top_k > 1:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-10
            topk_weights = topk_weights / denominator

        # [OPT] track expert load of current batch (for bias update)
        # use vectorized bincount instead of Python loop to avoid sync
        if self.training:
            self.expert_load = torch.bincount(
                topk_indices.flatten(),
                minlength=self.n_routed_experts
            ).to(dtype=torch.long)

        # calculate sequence-wise auxiliary loss (if alpha > 0)
        aux_seq_loss = None
        if self.seq_alpha > 0 and self.training:
            aux_seq_loss = self._compute_sequence_balance_loss(
                scores, topk_indices, bsz, seq_len
            )

        return topk_indices, topk_weights, aux_seq_loss

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
        # OPTIMIZED: Let autocast handle dtype - computation in native precision
        scores_reshaped = scores.view(bsz, seq_len, self.n_routed_experts)  # (bsz, seq_len, n_experts)
        topk_idx_reshaped = topk_idx.view(bsz, seq_len, self.top_k)         # (bsz, seq_len, top_k)

        # [OPT] OPTIMIZED: Use F.one_hot instead of scatter_ for better memory efficiency
        expert_mask = F.one_hot(
            topk_idx_reshaped.view(bsz, -1),  # (bsz, seq_len*top_k)
            num_classes=self.n_routed_experts # (bsz, seq_len*top_k, n_experts)
        ).to(scores.dtype)  # match input dtype

        # sum across tokens and normalize
        f_i = expert_mask.sum(dim=1) / (self.top_k * seq_len)  # (bsz, n_experts)

        # [OPT] vectorized P_i calculation: (bsz, n_experts):
        score_sums = scores_reshaped.sum(dim=2, keepdim=True)       # (bsz, seq_len, 1)
        normalized_scores = scores_reshaped / (score_sums + 1e-10)  # (bsz, seq_len, n_experts)
        P_i = normalized_scores.mean(dim=1)                         # (bsz, n_experts)

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
        if not self.training: return
        # calculate expected load per expert (uniform distribution)
        expected_load = (total_tokens * self.top_k) / self.n_routed_experts
        # vectorized bias update: compute difference and update all biases at once
        load_diff = self.expert_load.float() - expected_load
        self.expert_bias -= torch.sign(load_diff) * self.bias_update_speed


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
        # use FusedSharedExperts if shared experts are more than 1
        if n_shared_experts > 1:
            self.shared_experts = FusedSharedExperts(
                d_model=d_model,
                d_ff=d_ff,
                n_shared_experts=n_shared_experts,
                device=device,
                dtype=dtype
            )
        elif n_shared_experts == 1:
            self.shared_experts = MLP(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        identity = x
        batch_size, seq_len, _ = x.shape

        # choose expert using gate mechanism (auxiliary sequence-wise loss)
        topk_idx, topk_weight, aux_seq_loss = self.gate(x)
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
            sorted_token_idx = token_indices[sorted_expert_idx]   # which original token
            sorted_weights = flat_topk_weight[sorted_expert_idx]  # corresponding weights

            # permute input tokens to match expert grouping (contiguous memory access)
            sorted_tokens = x_flat[sorted_token_idx]  # (batch*seq*top_k, d_model)

            # count how many tokens each expert processes (n_routed_experts,)
            expert_token_counts = torch.bincount(
                flat_topk_idx, minlength=self.n_routed_experts
            )

            # compute cumulative offsets: [0, count[0], count[0]+count[1], ...]
            token_offsets = torch.cat([
                torch.tensor([0], device=x_flat.device, dtype=torch.long),
                torch.cumsum(expert_token_counts, dim=0)
            ])

            y_sorted = torch.zeros_like(sorted_tokens)  # allocate output buffer

            # handle each expert's chunk sequentially
            for expert_id in range(self.n_routed_experts):
                start_idx = token_offsets[expert_id]
                end_idx = token_offsets[expert_id + 1]

                # skip if no tokens assigned to this expert
                if start_idx == end_idx: continue

                # extract contiguous chunk (excellent cache locality!)
                expert_input = sorted_tokens[start_idx:end_idx]
                # single forward pass through expert
                expert_output = self.experts[expert_id](expert_input)
                # weight the expert output
                weights = sorted_weights[start_idx:end_idx].unsqueeze(-1)
                y_sorted[start_idx:end_idx] = expert_output * weights

            # un-sort the results to match original token order and accumulate
            # when multiple experts contribute to same token (top_k > 1)
            y_flat = torch.zeros(n_total_tokens, self.d_model, device=x_flat.device, dtype=x_flat.dtype)
            # expand sorted_token_idx for scatter operation
            sorted_token_idx_expanded = sorted_token_idx.unsqueeze(-1).expand_as(y_sorted)
            # scatter-add to accumulate contributions from multiple experts to same token
            y_flat.scatter_add(0, sorted_token_idx_expanded, y_sorted)

            y = y_flat.view(batch_size, seq_len, -1)
        else:
            # use existing optimized inference path
            y = self.moe_infer(x_flat, flat_topk_idx, topk_weight.view(-1, 1))
            y = y.view(batch_size, seq_len, -1)

        # apply fused shared experts and add to output
        if self.n_shared_experts > 0:
            y = y + self.shared_experts(identity)

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
        ''' optimized inference logic for Mixture-of-Experts '''
        expert_cache = torch.zeros_like(x)
        # sort indices to group tokens by expert
        idxs = flat_expert_indices.argsort()
        # calculate cumulative token counts per expert
        counts = flat_expert_indices.bincount().to(x.device)
        tokens_per_expert = torch.cumsum(counts, dim=0)

        token_idxs = idxs // self.num_experts_per_tok
        # loop through the batches of tokens for each expert
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            # get the batch of tokens for this expert
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            # process the batch and weight the output
            expert_out = expert(expert_tokens)
            # convert weights to match expert_out dtype
            weights = flat_expert_weights[idxs[start_idx:end_idx]].to(expert_out.dtype)
            expert_out = expert_out * weights  # use non-inplace op to avoid dtype issues
            # ensure expert_out matches expert_cache dtype before scatter_add_
            expert_out = expert_out.to(expert_cache.dtype)
            # scatter-add the results back to their original positions
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache
