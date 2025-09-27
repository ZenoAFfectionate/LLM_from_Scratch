import os
import json
import torch
import argparse

from cs336_basics.utils import softmax, load_checkpoint
from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.optimizer import AdamW
from cs336_basics.transformer import TransformerLM


@torch.no_grad()
def decode(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
) -> str:
    """
    Generate text from the model given a prompt.

    Args:
        model: Trained TransformerLM model.
        tokenizer: Tokenizer with encode() and decode().
        prompt (str): The starting text to condition on.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Temperature for softmax sampling.
        top_p (float): Nucleus sampling threshold (0 < p <= 1).
        device: Device to run on.

    Returns:
        Generated string including the prompt.
    """
    model.eval()
    model.to(device)

    # tokenize the prompt text into token index and convert to tensor
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # get the model's predictions for the next token
        logits = model(input_tensor)    # (1, seq_len, vocab_size)
        logits = logits[:, -1, :] / temperature  # (1, vocab_size)

        # apply top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)

        # remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = -float('Inf')

        # sample from the filtered distribution and append to the input tensor
        probs = softmax(logits, dim=-1)                                  # (1, vocab_size)
        next_token_id = torch.multinomial(probs, num_samples=1)          # (1, 1)
        input_tensor = torch.cat([input_tensor, next_token_id], dim=-1)  # (1, seq_len + 1)

    # decode the generated tokens back to string
    generated_ids = input_tensor.squeeze(0).tolist()
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

if __name__ == "__main__":
    console = Console()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate text using trained Transformer LM')
    parser.add_argument('--config', type=str, default='config/generate_tinystories.json',
                       help='Path to generation config JSON file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (overrides config)')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Input prompt for generation (if not provided, enters interactive mode)')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                       help='Maximum number of new tokens to generate (overrides config)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Temperature for sampling (overrides config)')
    parser.add_argument('--top_p', type=float, default=None,
                       help='Top-p threshold for nucleus sampling (overrides config)')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Override config with command line arguments if provided
    if args.checkpoint is not None:
        config['checkpoint_path'] = args.checkpoint
    if args.max_new_tokens is not None:
        config['max_new_tokens'] = args.max_new_tokens
    if args.temperature is not None:
        config['temperature'] = args.temperature
    if args.top_p is not None:
        config['top_p'] = args.top_p

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=config['vocab_file'],
        merges_filepath=config['merges_file'],
        special_tokens=config.get('special_tokens', ['<|endoftext|>'])
    )
    vocab_size = len(tokenizer.decoder_vocab)
    print(f"Vocabulary size: {vocab_size:,}")

    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta'],
        drop_p=config.get('dropout', None),
        device=device
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Load checkpoint
    print(f"Loading checkpoint: {config['checkpoint_path']}")
    if not os.path.exists(config['checkpoint_path']):
        raise FileNotFoundError(f"Checkpoint not found: {config['checkpoint_path']}")

    # Create a dummy optimizer for checkpoint loading 
    # (we don't need optimizer state for inference)
    dummy_optimizer = AdamW(model.parameters(), lr=1e-4)

    load_checkpoint(config['checkpoint_path'], model, dummy_optimizer)
    print("Checkpoint loaded successfully!")

    # Set model to evaluation mode
    model.eval()

    # -------------------
    # single prompt mode
    # -------------------
    if args.prompt is not None:
        print("\n" + "="*50)
        print(f"Prompt: {args.prompt}")
        print("="*50)
        print("Generating...\n")

        generated_text = decode(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=config['max_new_tokens'],
            temperature=config['temperature'],
            top_p=config['top_p'],
            device=device
        )
        print("--- Generated Text ---")
        print(generated_text)
        print("----------------------")

    # ----------------
    # Interactive mode
    # ----------------
    else:
        print("\n" + "="*50)
        print("🚀 Entering interactive mode. Type 'quit' or 'exit' to end.")
        print("="*50)
        while True:
            try:
                prompt = input("> ")
                if prompt.lower() in ["quit", "exit"]:
                    print("👋 Goodbye!")
                    break

                print("Generating...\n")
                generated_text = decode(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=config['max_new_tokens'],
                    temperature=config['temperature'],
                    top_p=config['top_p'],
                    device=device
                )
                print("--- Generated Text ---")
                print(generated_text)
                print("----------------------\n")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
