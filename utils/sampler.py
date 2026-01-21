import torch
import torch.nn as nn

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    Sampling parameters for text generation.

    Attributes:
        temperature: Temperature for sampling. Higher values make output more random.
                    Must be > 0 (greedy sampling not permitted).
        max_tokens: Maximum number of tokens to generate.
        ignore_eos: If True, continue generating even after EOS token.
    """
    temperature: float = 1.0
    max_tokens: int = 256
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"


class Sampler(nn.Module):
    """
    Sampler class for token sampling with temperature scaling.
    This is compatible with the vLLM-style inference engine.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """
        Sample next token from logits with per-sequence temperature scaling.

        Args:
            logits: Logits tensor of shape (batch_size, vocab_size)
            temperatures: Temperature tensor of shape (batch_size,)

        Returns:
            Sampled token IDs of shape (batch_size,)
        """
        # Convert to float and scale by temperature
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        # Sample tokens using Gumbel-max trick: argmax(log(p) + Gumbel(0,1))
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

    def __call__(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """Allow calling sampler instance directly like sampler(logits, temperatures)."""
        return self.forward(logits, temperatures)
