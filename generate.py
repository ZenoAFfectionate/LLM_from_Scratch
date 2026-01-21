import os
import torch
import argparse
import gradio as gr
from typing import List, Tuple

from model.config import Config
from utils.sampler import Sampler
from model.tokenizer.bpe_tokenizer import Tokenizer
from model.transformer import TransformerLM



@torch.inference_mode()
def generate(
    model: TransformerLM,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0
) -> List[List[int]]:
    """
    Generate new tokens based on the given prompt tokens using the specified model.
    Uses static graph approach with pre-allocated tensors and efficient KV caching.

    Args:
        model: The transformer model used for token generation.
        prompt_tokens: A list of lists containing the prompt tokens for each sequence.
        max_new_tokens: The maximum number of new tokens to generate.
        eos_id: The end-of-sequence token ID.
        temperature: The temperature value for sampling. Defaults to 1.0.
        top_p: Nucleus sampling threshold (0 < p <= 1.0). Defaults to 1.0 (disabled).

    Returns:
        List of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.context_length, \
        f"Prompt length exceeds model maximum sequence length (context_length={model.context_length})"

    # pre-allocate token tensor with static graph and fill in prompt tokens
    total_len = min(model.context_length, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    prev_pos = 0  # track generation state
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = (tokens != -1)  # distinguish prompt tokens from generated tokens

    # Create sampler instance for token sampling
    sampler = Sampler(temperature=temperature, top_p=top_p)

    # Unified prefill and decoding in generation loop
    for cur_pos in range(min(prompt_lens), total_len):
        # forward pass with incremental input (only new tokens since prev_pos)
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)  # use KV Cache

        if temperature > 0:
            next_token = sampler(logits[:, -1, :])
        else:
            next_token = logits[:, -1, :].argmax(dim=-1)

        # use prompt token if still in prompt range, otherwise use generated token
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token

        # update finished status: mark as finished if we generated eos_id (not from prompt)
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)

        prev_pos = cur_pos

        if finished.all(): break  # exit if all seqs are finished

    # Extract completion tokens (exclude prompt)
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        # truncate at eos_id if present
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)

    return completion_tokens


class ChatBot:
    """ChatBot wrapper for the Transformer Language Model with chat history support"""

    def __init__(self, model: TransformerLM, tokenizer: Tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate_response(
        self,
        message: str,
        history: List[Tuple[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_k: int = 32,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response for the current message with chat history context.

        Args:
            message: Current user message
            history: List of (user_message, bot_response) tuples
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: If > 0, only sample from top k tokens (0 = disabled)
            top_p: Nucleus sampling threshold

        Returns:
            Generated response string
        """
        context = ""  # build context from history
        for user_msg, bot_msg in history:
            context += f"User: {user_msg}\nAssistant: {bot_msg}\n"
        context += f"User: {message}\nAssistant:"

        # Tokenize the full context
        prompt_ids = self.tokenizer.encode(context)
        eos_id = self.tokenizer.special_tokens_ids[0] if self.tokenizer.special_tokens_ids else -1

        # Use the optimized generate function
        completion_tokens = generate(
            model=self.model,
            prompt_tokens=[prompt_ids],
            max_new_tokens=max_new_tokens,
            eos_id=eos_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # Decode the generated tokens
        response = self.tokenizer.decode(completion_tokens[0])

        return response.strip()


def load_model(config_path: str, checkpoint_override: str = None):
    """Load model and tokenizer from config"""

    # Load configuration using Config class
    config = Config.from_json(config_path)

    if checkpoint_override:
        config.checkpoint_path = checkpoint_override

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=config.vocab_file,
        merges_filepath=config.merges_file,
        special_tokens=config.special_tokens
    )
    vocab_size = len(tokenizer.decoder_vocab)
    print(f"Vocabulary size: {vocab_size:,}")

    # Update config with actual vocab_size
    config.vocab_size = vocab_size

    # Initialize model with Config object
    print("Initializing model...")
    model = TransformerLM(
        config=config,
        device=device
    ).to(device)

    # Print model configuration (simplified)
    attention_type = config.attention_type
    mlp_type = 'MoE' if config.use_moe else 'FFN'
    print(f"ATT Type: [{attention_type}]   MLP Type: [{mlp_type}]")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    print(f"Loading checkpoint: {config.checkpoint_path}")
    if not os.path.exists(config.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")

    # load checkpoint directly without optimizer (inference only)
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded successfully!")

    model.eval()

    return model, tokenizer, device, config.to_dict()


def create_gradio_interface(chatbot_instance: ChatBot, default_config: dict):
    """Create Gradio ChatInterface"""

    def respond(
        message: str,
        history: List[Tuple[str, str]],
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float
    ):
        """Wrapper function for Gradio ChatInterface"""
        if not message.strip():
            return ""

        response = chatbot_instance.generate_response(
            message=message,
            history=history,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        return response

    # Create the chat interface
    demo = gr.ChatInterface(
        fn=respond,
        title="🤖 Transformer Language Model Chat",
        description=f"Chat with a Transformer LM trained on {default_config.get('dataset', 'custom data')}. "
                   f"Model has {default_config.get('num_layers', 'N')} layers with GQA and MoE.",
        additional_inputs=[
            gr.Slider(
                minimum=10,
                maximum=500,
                value=default_config.get('max_new_tokens', 100),
                step=10,
                label="Max New Tokens",
                info="Maximum number of tokens to generate"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=default_config.get('temperature', 0.8),
                step=0.1,
                label="Temperature",
                info="Higher values make output more random"
            ),
            gr.Slider(
                minimum=0,
                maximum=100,
                value=default_config.get('top_k', 0),
                step=1,
                label="Top-k",
                info="If > 0, only sample from top k tokens (0 = disabled)"
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=default_config.get('top_p', 0.9),
                step=0.05,
                label="Top-p (Nucleus Sampling)",
                info="Cumulative probability threshold for sampling"
            ),
        ],
        examples=[
            ["Once upon a time, there was a"],
            ["Tell me a story about a brave knight"],
            ["What is the meaning of life?"],
            ["Write a poem about nature"],
        ],
        theme=gr.themes.Soft(),
        retry_btn="🔄 Regenerate",
        undo_btn="↩️ Undo",
        clear_btn="🗑️ Clear",
    )

    return demo


def main():
    parser = argparse.ArgumentParser(description='Web Interface for Transformer Language Model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/generate_tinystories.json',
        help='Path to generation config JSON file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public shareable link'
    )
    parser.add_argument(
        '--server_name',
        type=str,
        default='127.0.0.1',
        help='Server host address'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=7860,
        help='Server port'
    )
    args = parser.parse_args()

    print("="*60)
    print("🚀 Starting Transformer LM Web Interface")
    print("="*60)

    # Load model
    model, tokenizer, device, config = load_model(args.config, args.checkpoint)

    # Create chatbot instance
    chatbot = ChatBot(model, tokenizer, device)

    # Create Gradio interface
    demo = create_gradio_interface(chatbot, config)

    print("\n" + "="*60)
    print("✅ Interface ready! Launching web server...")
    print("="*60)

    # Launch the interface
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
