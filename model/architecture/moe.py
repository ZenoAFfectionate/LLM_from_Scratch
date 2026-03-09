import math
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.init as init
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
        # initialize gating weights for affinity score calculation
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size), device=device, dtype=dtype))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer('expert_load', torch.zeros(self.n_routed_experts, device=device, dtype=torch.long))
        self.register_buffer('expert_bias', torch.zeros(self.n_routed_experts, device=device, dtype=torch.float32))

    def forward(self, x):
        """ Forward pass with auxiliary-loss-free load balancing """
        bsz, seq_len, _ = x.shape

        # compute expert logits and scores
        x_flat = x.view(-1, x.shape[-1])
        logits = x_flat @ self.weight.t()
        scores = torch.sigmoid(logits)
        biased_scores = logits + self.expert_bias

        # select top-k experts based on biased logits and their unbiased weights
        _, topk_indices = torch.topk(biased_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weights = torch.gather(scores, dim=-1, index=topk_indices)
        if self.top_k > 1:  # re-normalize the weights of the selected
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-10
            topk_weights = topk_weights / denominator

        # use one_hot + sum instead of bincount: faster
        expert_token_counts = None
        if self.training:
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
        scores_reshaped = scores.view(bsz, seq_len, self.n_routed_experts)  # (bsz, seq_len, n_experts)
        topk_idx_reshaped = topk_idx.view(bsz, seq_len, self.top_k)         # (bsz, seq_len, top_k)

        expert_mask = F.one_hot(
            topk_idx_reshaped.view(bsz, -1),  # (bsz, seq_len*top_k)
            num_classes=self.n_routed_experts
        )  # (bsz, seq_len*top_k, n_experts), dtype=int64

        # compute f_i directly with int64 sum, then convert to float at the end
        expert_counts = expert_mask.sum(dim=1)  # (bsz, n_experts)
        f_i = expert_counts.to(scores.dtype) / (self.top_k * seq_len)  # (bsz, n_experts)

        # vectorized P_i calculation: (bsz, n_experts):
        score_sums = scores_reshaped.sum(dim=2, keepdim=True)       # (bsz, seq_len, 1)
        normalized_scores = scores_reshaped / (score_sums + 1e-10)  # (bsz, seq_len, n_experts)
        P_i = normalized_scores.mean(dim=1)  # (bsz, n_experts)

        seq_losses = (f_i * P_i).sum(dim=1)  # (bsz,)
        aux_seq_loss = self.seq_alpha * seq_losses.mean()

        return aux_seq_loss

    def update_bias(self, total_tokens):
        if not self.training: return
        with torch.no_grad():
            # calculate expected load per expert (uniform distribution)
            expected_load = (total_tokens * self.top_k) / self.n_routed_experts
            # vectorized bias update: compute difference and update at once
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
        capacity_factor: float = 1.5,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.capacity_factor = capacity_factor
        # ============================================== #
        # write efficient kernel for expert computation  #
        # ============================================== # 
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
            dtype=dtype,
        )
        # Initialize shared expert (single MLP)
        if n_shared_experts > 0:
            self.shared_expert = MLP(d_model, d_ff, device=device, dtype=dtype)

    @staticmethod
    def _compute_within_expert_positions(sorted_expert_ids):
        """
        Compute each token's position within its expert group, entirely on GPU.

        Given sorted expert assignments [0,0,0,2,2,5,5,5], returns within-expert
        positions [0,1,2,0,1,0,1,2]. Uses a cummax trick on segment boundaries
        to avoid any GPU-CPU synchronization.

        The trick: at each segment boundary (where expert ID changes), record the
        global position. cummax propagates this value forward through the segment.
        Subtracting from global position yields the local (within-segment) index.
        """
        n = sorted_expert_ids.shape[0]
        device = sorted_expert_ids.device
        global_pos = torch.arange(n, device=device)
        # Fuse boundary detection with position marking: at each segment boundary
        # (where expert ID changes), record the global position directly
        boundary_markers = torch.zeros(n, device=device, dtype=torch.long)
        if n > 1:
            mask = sorted_expert_ids[1:] != sorted_expert_ids[:-1]
            boundary_markers[1:] = mask * global_pos[1:]
        # cummax propagates the last boundary's global_pos forward,
        # so (global_pos - seg_starts) gives within-segment position
        seg_starts = torch.cummax(boundary_markers, dim=0).values
        return global_pos - seg_starts

    def forward(self, x):
        """
        MoE forward pass using padded fixed-capacity expert computation.

        Uses a fixed-capacity padding scheme to avoid GPU-CPU synchronization
        from variable-length torch.split(). All expert inputs are padded to a
        deterministic capacity derived from Python-int batch dimensions, enabling
        consistent tensor shapes and torch.compile compatibility.
        """
        identity = x
        batch_size, seq_len, _ = x.shape
        n_total_tokens = batch_size * seq_len

        # gate routing (aux_seq_loss and expert_token_counts are None when not training)
        topk_idx, topk_weight, aux_seq_loss, expert_token_counts = self.gate(x)
        x_flat = x.view(-1, x.shape[-1])   # (batch_size * seq_len, d_model)
        flat_topk_idx = topk_idx.view(-1)  # (batch_size * seq_len * top_k,)
        flat_topk_weight = topk_weight.view(-1)

        # create token indices for each expert selection to represents
        # which original token each selection belongs to (batch*seq*top_k,)
        token_indices = torch.arange(
            n_total_tokens, device=x_flat.device
        ).repeat_interleave(self.num_experts_per_tok)

        # sort by expert index to create contiguous chunks: O(N log N)
        sorted_expert_ids, sorted_expert_idx = torch.sort(flat_topk_idx)
        sorted_token_idx = token_indices[sorted_expert_idx]
        sorted_weight = flat_topk_weight[sorted_expert_idx]
        sorted_tokens = x_flat[sorted_token_idx]  # permute tokens to match expert group

        # compute within-expert position index on GPU using cummax trick
        within_pos = self._compute_within_expert_positions(sorted_expert_ids)

        # fixed capacity from Python ints only — no GPU-CPU sync!
        # With good load balancing, max tokens per expert ≈ N*top_k/n_experts.
        # capacity_factor provides headroom (e.g., 1.5 = 50% buffer).
        capacity = math.ceil(
            n_total_tokens * self.num_experts_per_tok
            / self.n_routed_experts * self.capacity_factor
        )
        within_pos = torch.clamp(within_pos, max=capacity - 1)

        # scatter tokens into padded buffer: (n_experts, capacity, d_model)
        padded_input = x_flat.new_zeros(
            self.n_routed_experts, capacity, self.d_model
        )
        padded_input[sorted_expert_ids, within_pos] = sorted_tokens

        # process all experts in-place (no separate output buffer needed)
        for expert_id in range(self.n_routed_experts):
            padded_input[expert_id] = self.experts[expert_id](padded_input[expert_id])

        # Gather outputs back in sorted order
        y_sorted = padded_input[sorted_expert_ids, within_pos]

        # Weight multiplication + scatter-add using segment reduction
        # This is MUCH faster than atomic scatter-add because:
        #   1. Re-sorts by target token to enable contiguous segment access
        #   2. Uses segment reduction - each output token sums its segment
        #   3. Fully coalesced reads and writes
        y_flat = fused_scatter_add_weighted(
            y_sorted,           # raw expert outputs
            sorted_token_idx,   # target token indices
            sorted_weight,      # routing weights
            n_total_tokens,     # number of original tokens
            self.num_experts_per_tok  # top_k for segment loop unrolling
        )

        y = y_flat.view(batch_size, seq_len, -1)
        # apply shared expert and add to output
        if self.n_shared_experts > 0:
            y = y + self.shared_expert(identity)

        # training-only bookkeeping (aux_seq_loss is None when not training)
        if self.training:
            self.aux_loss = aux_seq_loss
            self._total_tokens = n_total_tokens

        return y

    def update_expert_bias(self):
        """
        Update expert bias based on load balance.
        Should be called at the end of each training step.
        """
        if hasattr(self, '_total_tokens'):
            self.gate.update_bias(self._total_tokens)

