import atexit
from time import perf_counter
from tqdm.auto import tqdm
import torch.multiprocessing as mp

from model.config import Config
from model.tokenizer.bpe_tokenizer import Tokenizer
from engine.sampling_params import SamplingParams
from engine.sequence import Sequence
from engine.scheduler import Scheduler
from engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM inference engine that coordinates model execution, scheduling, and tokenization.

    This engine supports:
    - Single and multi-GPU inference with tensor parallelism
    - Continuous batching for efficient throughput
    - Custom BPE tokenizer or HuggingFace tokenizers
    """

    def __init__(self, config: Config | str, tokenizer: Tokenizer = None, **kwargs):
        """
        Initialize the LLM engine.

        Args:
            config: Either a Config object or path to a JSON config file
            tokenizer: Optional Tokenizer instance. If not provided, will attempt to
                      create from config (vocab_file, merges_file) or use a dummy tokenizer.
            **kwargs: Additional config overrides
        """
        # Load config from file if string path is provided
        if isinstance(config, str):
            config = Config.from_json(config)

        # Apply any config overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # Setup multiprocessing for tensor parallelism
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        tensor_parallel_size = getattr(config, 'tensor_parallel_size', 1)

        for i in range(1, tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # Initialize model runner on rank 0
        self.model_runner = ModelRunner(config, 0, self.events)

        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif hasattr(config, 'vocab_file') and config.vocab_file:
            # Use the custom BPE tokenizer
            self.tokenizer = Tokenizer.from_files(
                vocab_filepath=config.vocab_file,
                merges_filepath=config.merges_file,
                special_tokens=getattr(config, 'special_tokens', ["<|endoftext|>"])
            )
        else:
            # Try to use HuggingFace tokenizer if model path is provided
            model_path = getattr(config, 'checkpoint_path', None) or getattr(config, 'model', None)
            if model_path:
                try:
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                except Exception:
                    self.tokenizer = None
            else:
                self.tokenizer = None

        # Set EOS token ID
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, 'eos_token_id'):
                config.eos = self.tokenizer.eos_token_id
            elif hasattr(self.tokenizer, 'decoder_vocab'):
                # For custom BPE tokenizer, look for <|endoftext|>
                eos_token = "<|endoftext|>"
                config.eos = self.tokenizer.encoder_vocab.get(eos_token, -1)
            else:
                config.eos = -1
        else:
            config.eos = getattr(config, 'eos', -1)

        # Initialize scheduler
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        """Clean up resources and terminate worker processes."""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams = None):
        """
        Add a generation request to the queue.

        Args:
            prompt: Input text string or list of token IDs
            sampling_params: Sampling parameters (uses defaults if not provided)
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer not available. Please provide token IDs directly.")
            prompt = self.tokenizer.encode(prompt)

        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """Execute one inference step."""
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        """Check if all requests have been processed."""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> list[dict]:
        """
        Generate completions for a batch of prompts.

        Args:
            prompts: List of input strings or token ID lists
            sampling_params: Sampling parameters (single or per-prompt list)
            use_tqdm: Whether to show progress bar

        Returns:
            List of dicts with 'text' and 'token_ids' for each prompt
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        if sampling_params is None:
            sampling_params = SamplingParams()

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # Decode if tokenizer is available
        if self.tokenizer is not None:
            outputs = [
                {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
                for token_ids in outputs
            ]
        else:
            outputs = [{"text": None, "token_ids": token_ids} for token_ids in outputs]

        if use_tqdm:
            pbar.close()

        return outputs
