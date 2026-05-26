import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

from .utils import compute_varlen_positions  # only used to mirror sibling API


class _NaNScrub(torch.autograd.Function):
    """Replace NaN/Inf with 0 in both forward output AND backward gradient.

    Workaround for the fla 0.4.2 `chunk_gated_delta_rule` kernel, whose
    forward and backward Triton kernels both allocate their output buffers
    via `torch.empty()` and on some input shapes leave a subset of positions
    unwritten — those positions contain whatever the CUDA allocator handed
    out (commonly NaN/Inf from earlier tensors). Symptom: an alternating
    clean/dirty NaN pattern across repeated calls with identical inputs.
    Scrubbing on both sides keeps the rest of the network finite.
    """

    @staticmethod
    def forward(ctx, x):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nan_to_num(grad_output, nan=0.0, posinf=0.0, neginf=0.0)


def _scrub_nan(x: torch.Tensor) -> torch.Tensor:
    return _NaNScrub.apply(x)


class GatedDeltaAttention(nn.Module):
    """
    Gated Delta Attention (GDA) as used in Qwen3-Next's linear-attention layers.

    Reference: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
    (Yang et al., 2024) and the Qwen3-Next technical report (§3.2).

    Architecture (per layer):
        1.  Fused input projection      hidden -> [q | k | v | z]
            Auxiliary projection        hidden -> [b | a]
        2.  Depth-wise short conv (kernel=4, SiLU) over the concatenation of q/k/v
        3.  L2-normalise q, k per head (fused inside the kernel)
        4.  Decay gate                  g_t = -exp(A_log) * softplus(a_t + dt_bias)
            Delta rate                  beta_t = sigmoid(b_t)
        5.  Gated delta-rule recurrence on the value-head dimension
                S_t = (I - beta_t * k_t k_t^T) * exp(g_t) * S_{t-1}
                      + beta_t * v_t * k_t^T
                o_t = S_t * q_t
        6.  Gated RMSNorm with z (FusedRMSNormGated, swish gate), output projection.

    The k dimension is *grouped* (num_v_heads = G * num_k_heads, GVA expansion),
    matching Qwen3-Next where num_v_heads > num_k_heads.

    This module mirrors the GQA/MHA/MLA interface in this package:
        - forward(x):     training mode, full causal sequence
        - inference(x):   decode mode using a per-sequence recurrent cache
          (GDA has no KV-cache; it carries a (head_k_dim x head_v_dim) state
          and a (conv_kernel-1)-wide conv cache per layer per sequence)
    """

    def __init__(
        self,
        d_model: int,
        num_v_heads: int,
        num_k_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int = 4,
        norm_eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert num_v_heads % num_k_heads == 0, (
            "num_v_heads must be divisible by num_k_heads (GVA expansion)"
        )

        self.d_model = d_model
        self.num_v_heads = num_v_heads
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.conv_dim = self.key_dim * 2 + self.value_dim   # q | k | v

        # ---- fused projections (Qwen3-Next style, no bias) ----------------
        # qkvz: produces q (key_dim), k (key_dim), v (value_dim), z (value_dim)
        self.in_proj_qkvz = nn.Linear(
            d_model,
            2 * self.key_dim + 2 * self.value_dim,
            bias=False, device=device, dtype=dtype,
        )
        # ba: produces b (num_v_heads) and a (num_v_heads)
        self.in_proj_ba = nn.Linear(
            d_model, 2 * num_v_heads,
            bias=False, device=device, dtype=dtype,
        )

        # ---- depth-wise short convolution over (q|k|v) -------------------
        # ShortConvolution returns its updated conv-state when output_final_state
        # is True; this is what we cache during decode.
        self.conv1d = ShortConvolution(
            hidden_size=self.conv_dim,
            kernel_size=conv_kernel_size,
            activation="silu",          # SiLU as in Qwen3-Next
            bias=False,
            device=device, dtype=dtype,
        )

        # ---- decay parameters (per value-head) ---------------------------
        # A ~ U(1, 16), store its log so exp(A_log) > 0.
        # dt_bias ~ log-uniform(0.001, 0.1) — matches the Qwen3-Next reference
        # (transformers.models.qwen3_next.modeling_qwen3_next.Qwen3NextGatedDeltaNet).
        # With the previous constant `dt_bias = 1.0`, the decay
        #     g_t = -exp(A_log) * softplus(a_t + dt_bias)
        # spans [-25, -2] per head (A_log itself reaches log(16) ≈ 2.77 and
        # softplus(1+a) ≈ 1.3), which underflows the cumulative decay
        # exp(Σ g_t) in `chunk_gated_delta_rule`'s chunk recurrence and surfaces
        # as ~40% NaN output. The log-uniform init keeps g in roughly
        # [-1.7, -0.02], well within the kernel's numerically stable regime.
        A = torch.empty(num_v_heads, device=device).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt_bias_init = torch.empty(num_v_heads, device=device).uniform_(
            math.log(0.001), math.log(0.1)
        )
        self.dt_bias = nn.Parameter(dt_bias_init)
        self.dt_bias._no_weight_decay = True

        # ---- output gating norm + projection -----------------------------
        # FusedRMSNormGated applies: RMSNorm(x) * silu(gate)
        self.norm = FusedRMSNormGated(
            hidden_size=head_v_dim,
            elementwise_affine=True,
            eps=norm_eps,
            activation="swish",
            device=device, dtype=dtype,
        )
        self.out_proj = nn.Linear(
            self.value_dim, d_model,
            bias=False, device=device, dtype=dtype,
        )

        # ---- inference-mode state cache (set / cleared by the runner) ----
        # GDA does NOT use paged K/V cache, but we keep these attributes
        # for symmetry with the other attention modules.
        self.paged_attention = False
        self.k_cache = None
        self.v_cache = None
        self.block_size = 256

        # Recurrent state cache, populated lazily by `inference()`.
        # Shapes:
        #   recurrent_state : (batch, num_v_heads, head_k_dim, head_v_dim)
        #   conv_state      : (batch, conv_dim, conv_kernel_size - 1)
        self.recurrent_state: Optional[torch.Tensor] = None
        self.conv_state: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ helpers

    def _split_qkvz(self, qkvz: torch.Tensor):
        """Split fused projection output into (q, k, v, z)."""
        key_dim, value_dim = self.key_dim, self.value_dim
        q, k, v, z = torch.split(qkvz, [key_dim, key_dim, value_dim, value_dim], dim=-1)
        return q, k, v, z

    def _split_ba(self, ba: torch.Tensor):
        """Split fused projection output into (b, a)."""
        b, a = torch.split(ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        return b, a

    # =================================================================== train

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Training-mode forward over a full causal sequence.

        Args:
            x: (batch, seq_len, d_model)
            mask: ignored – GDA is intrinsically causal.

        Returns:
            (batch, seq_len, d_model)
        """
        bsz, seq_len, _ = x.shape

        # --- projections ---------------------------------------------------
        qkvz = self.in_proj_qkvz(x)
        ba = self.in_proj_ba(x)
        q, k, v, z = self._split_qkvz(qkvz)
        b, a = self._split_ba(ba)

        # --- short conv over (q | k | v) ----------------------------------
        mixed = torch.cat([q, k, v], dim=-1)            # (B, T, conv_dim)
        mixed, _ = self.conv1d(mixed, output_final_state=False)
        q, k, v = torch.split(
            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )

        # --- reshape into heads -------------------------------------------
        q = q.view(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.view(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.view(bsz, seq_len, self.num_v_heads, self.head_v_dim)

        # --- gates ---------------------------------------------------------
        # decay (log-space): g_t = -exp(A_log) * softplus(a_t + dt_bias)
        g = -torch.exp(self.A_log.float()) * F.softplus(
            a.float() + self.dt_bias.float()
        )                                                # (B, T, num_v_heads)
        beta = b.sigmoid()                              # (B, T, num_v_heads)

        # --- core: chunk-wise gated delta-rule recurrence -----------------
        # `use_qk_l2norm_in_kernel=True` performs the L2-norm of q/k inside
        # the kernel, matching the Qwen3-Next reference.
        out, _ = chunk_gated_delta_rule(
            q=q, k=k, v=v,
            g=g, beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )                                                # (B, T, num_v_heads, head_v_dim)
        # Scrub allocator garbage from kernel output AND its backward
        # gradient — fla 0.4.2's chunk Triton kernel leaves some positions
        # unwritten on certain input shapes. See `_NaNScrub` docstring above.
        out = _scrub_nan(out)

        # --- output gating + projection -----------------------------------
        # z is reshaped to (B, T, num_v_heads, head_v_dim) and used as the
        # gate input of FusedRMSNormGated.
        z = z.view(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        out = self.norm(out, z)                          # (B, T, HV, V)
        out = out.reshape(bsz, seq_len, self.value_dim)
        return self.out_proj(out)

    # =============================================================== inference

    def allocate_inference_cache(self, batch_size: int, device, dtype):
        """Allocate (or reset) the recurrent + conv state for `batch_size` seqs.

        - recurrent_state: stored in fp32 (state matrix is fp32 inside the kernel)
        - conv_state:      shape (B, conv_dim, kernel_size) — holds the last
                           `kernel_size` raw conv inputs (zero-initialised for
                           a fresh sequence, matching causal left-padding).
        """
        self.recurrent_state = torch.zeros(
            batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim,
            device=device, dtype=torch.float32,
        )
        self.conv_state = torch.zeros(
            batch_size, self.conv_dim, self.conv_kernel_size,
            device=device, dtype=dtype,
        )

    def reset_inference_cache(self):
        self.recurrent_state = None
        self.conv_state = None

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference forward.

        Unlike attention-based modules, GDA has no KV-cache; instead each
        sequence carries a (head_k_dim x head_v_dim) recurrent state and a
        (conv_kernel-1)-wide conv state. Two regimes are supported:

        - prefill (seq_len > 1 or no state yet): full chunk-wise pass over
          the prompt, then save the final recurrent / conv state.
        - decode  (seq_len == 1 with state): single-step recurrent update via
          `fused_recurrent_gated_delta_rule`.

        The caller (ModelRunner) is responsible for calling
        `allocate_inference_cache` once per request and clearing it between
        requests.
        """
        bsz, seq_len, _ = x.shape
        device, dtype = x.device, x.dtype

        # ---- look at context if it exists (mirrors sibling modules) ------
        try:
            from utils.context import get_context
            context = get_context()
        except Exception:
            context = None

        is_prefill = (
            context.is_prefill if context is not None and hasattr(context, "is_prefill")
            else (self.recurrent_state is None or seq_len > 1)
        )

        if self.recurrent_state is None or self.recurrent_state.shape[0] != bsz:
            self.allocate_inference_cache(bsz, device, dtype)

        # --- projections (identical to training path) ---------------------
        qkvz = self.in_proj_qkvz(x)
        ba = self.in_proj_ba(x)
        q, k, v, z = self._split_qkvz(qkvz)
        b, a = self._split_ba(ba)

        mixed = torch.cat([q, k, v], dim=-1)

        # --- short conv with state update ---------------------------------
        # On the very first call (no cache yet) pass cache=None so
        # ShortConvolution allocates the correct shape; subsequent calls
        # carry the (B, conv_dim, kernel_size) cache forward.
        mixed, new_conv_state = self.conv1d(
            mixed,
            cache=self.conv_state,
            output_final_state=True,
        )
        self.conv_state = new_conv_state

        q, k, v = torch.split(
            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        q = q.view(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.view(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.view(bsz, seq_len, self.num_v_heads, self.head_v_dim)

        g = -torch.exp(self.A_log.float()) * F.softplus(
            a.float() + self.dt_bias.float()
        )
        beta = b.sigmoid()

        if is_prefill:
            out, final_state = chunk_gated_delta_rule(
                q=q, k=k, v=v,
                g=g, beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            # See training-path comment: scrub allocator garbage from kernel output.
            out = _scrub_nan(out)
        else:
            # single-token step
            out, final_state = fused_recurrent_gated_delta_rule(
                q=q, k=k, v=v,
                g=g, beta=beta,
                initial_state=self.recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        self.recurrent_state = final_state

        z = z.view(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        out = self.norm(out, z)
        out = out.reshape(bsz, seq_len, self.value_dim)
        return self.out_proj(out)
