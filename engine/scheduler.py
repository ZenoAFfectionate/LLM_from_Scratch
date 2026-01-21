from collections import deque

from model.config import Config
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager


class Scheduler:
    """
    The Scheduler coordinates the lifecycle of requests. It decides which 
    sequences should be processed in the next engine iterationbased on 
    hardware constraints (max tokens, max sequences) and memory availability.
    """

    def __init__(self, config: Config):
        self.eos = config.eos
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # memory block manager for KV cache allocation:
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()  # sequence waiting to be scheduled
        self.running: deque[Sequence] = deque()  # sequence currently running

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Determines the batch of sequences to run in the next step."""
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # =============
        # prefill stage
        # =============
        # try to move sequences from 'waiting' to 'running'
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # check if we can allocate memory and tokens for this sequence
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        # if any sequences were scheduled, return them
        if scheduled_seqs: return scheduled_seqs, True

        # ============
        # decode stage
        # ============
        # continue generating token for running sequences
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # check if we can allocate new block for this sequence
            while not self.block_manager.can_append(seq):
                # if memory is full, preempt another sequence
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # memory is available for moving forward
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        # consistency check and re-populate running queue
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        Evict a sequence from GPU memory to free up blocks. The sequence
        is moved back to the waiting queue to be resumed later.
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)   # release all physical blocks
        self.waiting.appendleft(seq)         # move seq back to waiting queue

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """ Update sequences after the model generates new tokens """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # check if sequence is finished: reach the end or max tokens
            if (not seq.ignore_eos and token_id == self.eos) or \
               seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                # release memory blocks
                self.block_manager.deallocate(seq)
                # remove from running queue
                self.running.remove(seq)
