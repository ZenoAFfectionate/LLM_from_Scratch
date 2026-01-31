"""
Configuration class for Transformer Language Model.

This module provides a unified configuration system to eliminate parameter redundancy
across different modules. The Config class supports JSON serialization/deserialization
and provides sensible defaults for training with MLA+MoE on OpenWebText.
"""

import json
from typing import List, Optional, Union
from pathlib import Path


class Config:
    """
    Configuration class for Transformer Language Model.

    This class centralizes all model, training, and data configuration parameters,
    replacing the need for passing numerous individual parameters across modules.

    Key features:
    - Supports loading from JSON files via from_json() class method
    - Supports saving to JSON files via save() method
    - Provides sensible defaults for MLA+MoE training on OpenWebText
    - All attributes are accessible as object properties

    Model Architecture Parameters:
        vocab_size: Vocabulary size (determined by tokenizer)
        context_length: Maximum sequence length (default: 2048)
        d_model: Model dimension (default: 768)
        num_layers: Number of transformer layers (default: 12)
        num_heads: Number of attention heads (default: 16)
        d_ff: Feed-forward dimension (default: 3072)
        dropout: Dropout probability (default: 0.1)

    Attention Configuration:
        attention_type: Type of attention mechanism ("MHA", "GQA", "MLA", default: "MLA")
        num_kv_heads: Number of key-value heads for GQA (default: None, uses num_heads)
        rope_theta: RoPE theta parameter (default: 10000.0)
        rope_dim: Dimension for RoPE in MLA (default: 16)
        q_lora_rank: Query LoRA rank for MLA (default: 128)
        kv_lora_rank: KV LoRA rank for MLA (default: 256)

    MoE Configuration:
        use_moe: Whether to use Mixture of Experts (default: True)
        moe_layers: List of layer indices to apply MoE, None = all layers (default: None)
        n_routed_experts: Number of routed experts (default: 8)
        num_experts_per_tok: Number of experts per token (default: 1)
        n_shared_experts: Number of shared experts (default: 1)
        aux_seq_loss_alpha: Auxiliary loss weight for load balancing (default: 0.01)
        bias_update_speed: Speed of bias update for load balancing (default: 0.01)

    Engram Configuration:
        use_engram: Whether to use Engram module (default: False)
        engram_layer_ids: List of layer indices to apply Engram (default: [1, 5])
        engram_max_ngram_size: Maximum n-gram order (default: 3)
        engram_n_embed_per_ngram: Embedding dimension per n-gram (default: 512)
        engram_n_head_per_ngram: Number of hash heads per n-gram (default: 8)
        engram_vocab_size: Vocab size per n-gram order (default: [10007, 10009])
        engram_kernel_size: Kernel size for short convolution (default: 4)
        engram_pad_id: Padding token ID for Engram (default: 2)
        engram_tokenizer_path: Path to tokenizer for Engram compression (default: None)
        hc_mult: Hyper-connection multiplicity (default: 1)

    Training Configuration:
        batch_size: Training batch size (default: 8)
        max_iterations: Maximum training iterations (default: 50000)
        max_lr: Maximum learning rate (default: 2e-4)
        min_lr: Minimum learning rate (default: 2e-5)
        warmup_iterations: Number of warmup iterations (default: 2500)
        beta1: Adam beta1 parameter (default: 0.9)
        beta2: Adam beta2 parameter (default: 0.999)
        eps: Adam epsilon parameter (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.1)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        use_amp: Whether to use automatic mixed precision (default: True)

    Data Configuration:
        dataset: Dataset name (default: "OpenWebText")
        data_dir: Directory containing the data (default: "./data/OpenWebText")
        vocab_file: Path to vocabulary file (default: None)
        merges_file: Path to BPE merges file (default: None)
        special_tokens: List of special tokens (default: ["<|endoftext|>"])

    Logging and Checkpointing:
        run_name: Name of the training run (default: "transformer_training")
        log_interval: Logging interval in iterations (default: 100)
        eval_interval: Evaluation interval in iterations (default: 1000)
        eval_batches: Number of batches for evaluation (default: 1600)
        checkpoint_dir: Directory to save checkpoints (default: "./checkpoints")

    Other:
        seed: Random seed (default: 42)
        num_workers: Number of data loading workers (default: 8)
    """

    def __init__(
        self,
        # Model architecture
        vocab_size: int = 32000,
        context_length: int = 2048,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 16,
        d_ff: int = 3072,
        dropout: float = 0.1,

        # Attention configuration
        attention_type: str = "MLA",
        num_kv_heads: Optional[int] = None,
        rope_theta: float = 10000.0,
        rope_dim: Optional[int] = 16,
        q_lora_rank: Optional[int] = 128,
        kv_lora_rank: Optional[int] = 256,

        # MoE configuration
        use_moe: bool = True,
        moe_layers: Optional[List[int]] = None,
        n_routed_experts: int = 8,
        num_experts_per_tok: int = 1,
        n_shared_experts: int = 1,
        aux_seq_loss_alpha: float = 0.01,
        bias_update_speed: float = 0.01,

        # Engram configuration
        use_engram: bool = False,
        engram_layer_ids: Optional[List[int]] = None,
        engram_max_ngram_size: int = 3,
        engram_n_embed_per_ngram: int = 512,
        engram_n_head_per_ngram: int = 8,
        engram_vocab_size: Optional[List[int]] = None,
        engram_kernel_size: int = 4,
        engram_pad_id: int = 2,
        engram_tokenizer_path: Optional[str] = None,
        hc_mult: int = 1,

        # Training configuration
        batch_size: int = 8,
        max_iterations: int = 50000,
        max_lr: float = 2e-4,
        min_lr: float = 2e-5,
        warmup_iterations: int = 2500,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,

        # Data configuration
        dataset: str = "OpenWebText",
        data_dir: str = "./data/OpenWebText",
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        special_tokens: List[str] = None,

        # Logging and checkpointing
        run_name: str = "transformer_training",
        log_interval: int = 100,
        eval_interval: int = 1000,
        eval_batches: int = 1600,
        checkpoint_dir: str = "./checkpoints",

        # Other
        seed: int = 42,
        num_workers: int = 8,

        # Additional fields from config files
        checkpoint_path: Optional[str] = None,
        train_file: Optional[str] = None,
        valid_file: Optional[str] = None,
        **kwargs
    ):
        """Initialize configuration with default values for MLA+MoE on OpenWebText."""

        # Model architecture
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        # Attention configuration
        self.attention_type = attention_type
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.rope_theta = rope_theta
        self.rope_dim = rope_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        # MoE configuration
        self.use_moe = use_moe
        self.moe_layers = moe_layers
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.aux_seq_loss_alpha = aux_seq_loss_alpha
        self.bias_update_speed = bias_update_speed

        # Engram configuration
        self.use_engram = use_engram
        self.engram_layer_ids = engram_layer_ids if engram_layer_ids is not None else [1, 5]
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_n_embed_per_ngram = engram_n_embed_per_ngram
        self.engram_n_head_per_ngram = engram_n_head_per_ngram
        self.engram_vocab_size = engram_vocab_size if engram_vocab_size is not None else [10007, 10009]
        self.engram_kernel_size = engram_kernel_size
        self.engram_pad_id = engram_pad_id
        self.engram_tokenizer_path = engram_tokenizer_path
        self.hc_mult = hc_mult

        # Training configuration
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iterations = warmup_iterations
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp

        # Data configuration
        self.dataset = dataset
        self.data_dir = data_dir
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.special_tokens = special_tokens if special_tokens is not None else ["<|endoftext|>"]

        # Logging and checkpointing
        self.run_name = run_name
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_batches = eval_batches
        self.checkpoint_dir = checkpoint_dir

        # Other
        self.seed = seed
        self.num_workers = num_workers

        # Additional fields
        self.checkpoint_path = checkpoint_path
        self.train_file = train_file
        self.valid_file = valid_file

        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from a JSON file.

        Args:
            json_path: Path to the JSON configuration file

        Returns:
            Config instance with parameters loaded from the file

        Example:
            config = Config.from_json('config/train_openwebtext.json')
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")

        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def save(self, json_path: Union[str, Path]):
        """
        Save configuration to a JSON file.

        Args:
            json_path: Path where the JSON file will be saved

        Example:
            config.save('config/my_config.json')
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def to_dict(self) -> dict:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary containing all configuration parameters
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self) -> str:
        """String representation of the configuration."""
        config_str = "Config(\n"
        for key, value in sorted(self.to_dict().items()):
            config_str += f"  {key}={repr(value)},\n"
        config_str += ")"
        return config_str

    def update(self, **kwargs):
        """
        Update configuration parameters.

        Args:
            **kwargs: Parameters to update

        Example:
            config.update(batch_size=16, max_lr=3e-4)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")
