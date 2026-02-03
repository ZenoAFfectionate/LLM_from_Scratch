import os
import json
import argparse

from model.tokenizer.bpe_tokenizer import Tokenizer
from engine import LLMEngine
from utils.sampler import SamplingParams


def build_engine_config(config: dict, tokenizer: Tokenizer) -> dict:
    """
    Build engine configuration dictionary from config file.

    Args:
        config: Original configuration dictionary
        tokenizer: Tokenizer instance (for vocab_size and eos token id)

    Returns:
        Engine configuration dictionary
    """
    vocab_size = len(tokenizer.decoder_vocab)

    # Get EOS token ID
    eos_token_id = tokenizer.special_tokens_ids[0] if tokenizer.special_tokens_ids else vocab_size - 1

    engine_config = {
        # Model config
        'vocab_size': vocab_size,
        'max_model_length': config.get('context_length', 2048),
        'hidden_size': config.get('d_model', 768),
        'num_layers': config.get('num_layers', 12),
        'num_heads': config.get('num_heads', 16),
        'num_kv_heads': config.get('num_kv_heads', config.get('num_heads', 16)),
        'intermediate_size': config.get('d_ff', 3072),
        'attention_type': config.get('attention_type', 'GQA'),
        'rope_theta': config.get('rope_theta', 10000.0),
        'rope_dim': config.get('rope_dim', None),
        'q_lora_rank': config.get('q_lora_rank', None),
        'kv_lora_rank': config.get('kv_lora_rank', None),
        'use_moe': config.get('use_moe', False),
        'n_routed_experts': config.get('n_routed_experts', 8),
        'num_experts_per_tok': config.get('num_experts_per_tok', 1),
        'n_shared_experts': config.get('n_shared_experts', 1),
        'checkpoint_path': config.get('checkpoint_path'),

        # Engine config
        'world_size': config.get('world_size', 1),
        'max_num_sequences': config.get('max_num_sequences', 16),
        'max_num_batched_tokens': config.get('max_num_batched_tokens', 2048),
        'max_cached_blocks': config.get('max_cached_blocks', 512),
        'block_size': config.get('block_size', 256),
        'eos': eos_token_id,
        'max_num_seqs': config.get('max_num_seqs', 16),
        'max_num_batch_tokens': config.get('max_num_batch_tokens', 4096),
        'gpu_memory_utilization': config.get('gpu_memory_utilization', 0.9),
        'enforce_eager': config.get('enforce_eager', False),
    }

    return engine_config


def main():
    parser = argparse.ArgumentParser(description='TransformerLM Inference with LLMEngine')
    parser.add_argument('--config', type=str, default='config/[MLA+MoE]generate_openwebtext.json', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (overrides config)')
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch'], default='interactive', help='Inference mode: interactive or batch')
    parser.add_argument('--prompts', type=str, nargs='+', default=None, help='List of prompts for batch mode')
    parser.add_argument('--prompts_file', type=str, default=None, help='File containing prompts (one per line) for batch mode')
    parser.add_argument('--output_file', type=str, default=None, help='Output file path for batch mode results')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max tokens to generate (overrides config)')
    parser.add_argument('--temperature', type=float, default=None, help='Sampling temperature (overrides config)')
    args = parser.parse_args()

    # Load config
    print("=" * 60)
    print("TransformerLM Inference with LLMEngine")
    print("=" * 60)
    print(f"Loading config from: {args.config}")

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Override checkpoint path
    if args.checkpoint:
        config['checkpoint_path'] = args.checkpoint

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=config['vocab_file'],
        merges_filepath=config['merges_file'],
        special_tokens=config.get('special_tokens', [])
    )
    vocab_size = len(tokenizer.decoder_vocab)
    print(f"Vocabulary size: {vocab_size:,}")

    # Build engine config
    engine_config = build_engine_config(config, tokenizer)

    # Print model configuration
    print(f"\nModel Configuration:")
    print(f"  - Attention Type: {config.get('attention_type', 'GQA')}")
    print(f"  - MLP Type: {'MoE' if config.get('use_moe', False) else 'FFN'}")
    print(f"  - Num Layers: {config.get('num_layers', 12)}")
    print(f"  - Hidden Size: {config.get('d_model', 768)}")
    print(f"  - Context Length: {config.get('context_length', 2048)}")
    print(f"  - Checkpoint: {config.get('checkpoint_path', 'N/A')}")

    # Initialize Engine
    print("\nInitializing LLMEngine...")
    engine = LLMEngine(engine_config, tokenizer=tokenizer)
    print("Engine initialized successfully!\n")

    # Configure sampling parameters
    temperature = args.temperature if args.temperature else config.get('temperature', 0.8)
    max_tokens = args.max_tokens if args.max_tokens else config.get('max_new_tokens', 256)

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        ignore_eos=False,
        max_model_length=config.get('context_length', 2048)
    )

    print(f"Sampling Parameters:")
    print(f"  - Temperature: {temperature}")
    print(f"  - Max Tokens: {max_tokens}")

    # Execute inference based on mode
    if args.mode == 'interactive':
        print("\n" + "=" * 60)
        print("Interactive Inference Mode")
        print("Type 'quit' or 'exit' to exit")
        print("=" * 60 + "\n")

        while True:
            try:
                prompt = input("User: ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ['quit', 'exit']:
                    print("Exiting interactive mode...")
                    break

                output = engine.generate([prompt], sampling_params)
                generated_text = output['text'][0] if output['text'] else ""
                print(f"Assistant: {generated_text}\n")

            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error during generation: {e}")
    else:
        # Batch mode
        prompts = []
        if args.prompts:
            prompts = args.prompts
        elif args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # Use default test prompts
            prompts = [
                "Once upon a time,",
                "The meaning of life is",
                "In a world where technology"
            ]
            print("\nNo prompts provided, using default test prompts.")

        print(f"\nProcessing {len(prompts)} prompts...")
        print("-" * 60)

        output = engine.generate(prompts, sampling_params)

        results = []
        for i, (prompt, generated) in enumerate(zip(prompts, output['text'])):
            result = {
                'prompt': prompt,
                'generated': generated,
                'token_ids': output['token_ids'][i]
            }
            results.append(result)

            print(f"\n[Prompt {i+1}]")
            print(f"Input: {prompt}")
            print(f"Output: {generated}")

        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.output_file}")

    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
  