import torch
import torch.nn as nn

from model.utils import RMSNorm


class MTPModule(nn.Module):
    """
    Multi-Token Prediction (MTP) Module for sequential prediction of future tokens.

    Based on DeepSeek-V3 paper: sequentially predicts additional tokens while maintaining
    the complete causal chain at each prediction depth.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        rope: Rotary position embedding instance
        drop_p: Dropout probability
        use_moe: Whether to use MoE in this module
        n_routed_experts: Number of routed experts (if MoE)
        num_experts_per_tok: Number of experts per token (if MoE)
        n_shared_experts: Number of shared experts (if MoE)
        aux_seq_loss_alpha: Auxiliary loss alpha (if MoE)
        num_kv_heads: Number of KV heads for GQA
        bias_update_speed: Bias update speed (if MoE)
        device: Device to place the module
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope,
        drop_p: float,
        use_moe: bool,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_shared_experts: int,
        aux_seq_loss_alpha: float,
        num_kv_heads: int,
        attention_type: str = "GQA",
        d_rope: int = None,
        kv_lora_rank: int = None,
        q_lora_rank: int = None,
        bias_update_speed: float = 0.01,
        device=None
    ):
        super().__init__()

        # Import here to avoid circular dependency
        from model.transformer import TransformerBlock

        # Projection matrix M_k: projects [h_{k-1}; Emb(t_{i+k})] from 2d to d
        # Input: [RMSNorm(h_{k-1}); RMSNorm(Emb(t_{i+k}))] with shape (2*d_model)
        # Output: h'_k with shape (d_model)
        self.projection = nn.Linear(2 * d_model, d_model, device=device)

        # Transformer block at this depth
        self.transformer_block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            rope=rope,
            drop_p=drop_p,
            use_moe=use_moe,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_shared_experts=n_shared_experts,
            aux_seq_loss_alpha=aux_seq_loss_alpha,
            num_kv_heads=num_kv_heads,
            attention_type=attention_type,
            d_rope=d_rope,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            bias_update_speed=bias_update_speed,
            device=device
        )

        # RMSNorm layers for input normalization
        self.norm_h = RMSNorm(d_model, device=device)
        self.norm_emb = RMSNorm(d_model, device=device)

    def forward(
        self,
        h_prev: torch.Tensor,          # (batch_size, seq_len, d_model)
        token_embeddings: torch.Tensor  # (batch_size, seq_len, d_model)
    ) -> torch.Tensor:
        """
        Forward pass of MTP module.

        Args:
            h_prev: Representations from previous depth, shape (batch, seq_len, d_model)
            token_embeddings: Embeddings of future tokens, shape (batch, seq_len, d_model)

        Returns:
            h_k: Representations at current depth, shape (batch, seq_len, d_model)
        """
        # Normalize both inputs
        h_prev_norm = self.norm_h(h_prev)                 # (batch, seq_len, d_model)
        token_emb_norm = self.norm_emb(token_embeddings)  # (batch, seq_len, d_model)

        # Concatenate along feature dimension
        combined = torch.cat([h_prev_norm, token_emb_norm], dim=-1)  # (batch, seq_len, 2*d_model)

        # Project to d_model dimension
        h_prime = self.projection(combined)  # (batch, seq_len, d_model)

        # Pass through transformer block (no cache for MTP during training)
        h_k, _ = self.transformer_block(h_prime, kv_cache=None, use_cache=False)

        return h_k


class MultiTokenPredictor(nn.Module):
    """
    Multi-Token Prediction (MTP) wrapper that manages D sequential MTP modules.

    Args:
        num_depths: Number of additional prediction depths (D in paper)
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        rope: Rotary position embedding instance
        drop_p: Dropout probability
        use_moe: Whether to use MoE
        n_routed_experts: Number of routed experts
        num_experts_per_tok: Number of experts per token
        n_shared_experts: Number of shared experts
        aux_seq_loss_alpha: Auxiliary loss alpha
        num_kv_heads: Number of KV heads for GQA
        bias_update_speed: Bias update speed
        device: Device to place modules
    """
    def __init__(
        self,
        num_depths: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope,
        drop_p: float,
        use_moe: bool,
        moe_layers: list,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_shared_experts: int,
        aux_seq_loss_alpha: float,
        num_kv_heads: int,
        attention_type: str = "GQA",
        d_rope: int = None,
        kv_lora_rank: int = None,
        q_lora_rank: int = None,
        bias_update_speed: float = 0.01,
        device=None
    ):
        super().__init__()
        self.num_depths = num_depths

        # Create D MTP modules
        self.mtp_modules = nn.ModuleList([
            MTPModule(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope=rope,
                drop_p=drop_p,
                use_moe=(use_moe and (moe_layers is None or i in moe_layers)),
                n_routed_experts=n_routed_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_shared_experts=n_shared_experts,
                aux_seq_loss_alpha=aux_seq_loss_alpha,
                num_kv_heads=num_kv_heads,
                attention_type=attention_type,
                d_rope=d_rope,
                kv_lora_rank=kv_lora_rank,
                q_lora_rank=q_lora_rank,
                bias_update_speed=bias_update_speed,
                device=device
            )
            for i in range(num_depths)
        ])

    def forward(
        self,
        h_main: torch.Tensor,           # Main model representations
        token_ids: torch.Tensor,        # Input token IDs
        embedding_layer: nn.Module,     # Shared embedding layer
    ):
        """
        Forward pass through all MTP depths.

        Args:
            h_main: Main model output representations, shape (batch, seq_len, d_model)
            token_ids: Input token IDs, shape (batch, seq_len)
            embedding_layer: Shared embedding layer from main model

        Returns:
            List of representations at each depth: [h_1, h_2, ..., h_D]
        """
        batch_size, seq_len, d_model = h_main.shape

        # Store representations at each depth
        depth_representations = []

        h_prev = h_main  # h_0 is the main model output

        for k, mtp_module in enumerate(self.mtp_modules):
            # Get embeddings for tokens at offset k+1
            # For k-th depth, we need tokens at positions [1+k, 2+k, ..., T-1+k]
            # These correspond to token_ids[:, k+1:T] where T = seq_len

            if k + 1 >= seq_len:
                # Not enough tokens for this depth
                break

            # Get future token embeddings: tokens at positions i+k+1
            future_token_ids = token_ids[:, k+1:]  # (batch, seq_len - k - 1)
            future_token_embeddings = embedding_layer(future_token_ids)  # (batch, seq_len-k-1, d_model)

            # Slice h_prev to match the sequence length
            h_prev_sliced = h_prev[:, :seq_len - k - 1, :]  # (batch, seq_len - k - 1, d_model)

            # Forward through MTP module
            h_k = mtp_module(h_prev_sliced, future_token_embeddings)

            depth_representations.append(h_k)
            h_prev = h_k

        return depth_representations
