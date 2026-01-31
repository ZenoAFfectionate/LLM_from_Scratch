import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from sequence import Sequence
from scheduler import Scheduler
from model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    import os
    import sys
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)  # Line buffering
    # create ModelRunner instance and enter loop
    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    """
    LLM Engine that manages multiple processes for model inference.
    """
    def __init__(self, config: dict):
        # initialize scheduler
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256)
        )
        # initialize worker processes (rank 0)
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        # traverse ranks and initialize each process
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        atexit.register(self.exit)


    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()


    def step(self) -> tuple[list[int], bool]:
        '''A single step of scheduling, model execution, and postprocessing'''
        # schedule sequences
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        if not scheduled_sequences: return [], is_prefill
        # run the model
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        # postprocess the outputs
        self.scheduler.postprocess(scheduled_sequences, outputs)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)
        return outputs, num_processed_tokens, is_prefill


    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        '''Add a prompt string to the waiting queue by first transforming it to a Sequence object.'''
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), sampling_params=sampling_params))


    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        '''Perform generation for a list of requests (serve as prompts)'''
        # add prompts to the waiting queue
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}
        # call step until all sequences are finished
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})
        # sort generated tokens by seq_id to maintain order
        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output
