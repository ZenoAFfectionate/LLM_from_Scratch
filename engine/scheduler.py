from collections import deque
from engine.sequence import Sequence, SequenceStatus
from engine.block_manager import BlockManager


class Scheduler:
    """Scheduler for managing and scheduling sequences for generation."""
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        # initialize block manager and set max parameters
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens  # max tokens per batch
        self.max_num_sequences = max_num_sequences            # max sequences per batch
        # initialize sequence queues
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool]:
        """Schedule sequences for the next generation step."""
        scheduled_sequences = []      # sequences scheduled for this step
        current_scheduled_tokens = 0  # current total tokens scheduled

        # try to schedule for prefilling from waiting queue if not exceeding
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            if (self.block_manager.can_allocate(seq) and
                len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens):
                seq = self.waiting.popleft()      # remove from waiting queue
                self.block_manager.allocate(seq)  # allocate blocks for sequence
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)          # add to running queue
                scheduled_sequences.append(seq)   # schedule to prefilling
                current_scheduled_tokens += len(seq)
            else:
                break
        # return any sequences scheduled for prefilling
        if scheduled_sequences:
            return scheduled_sequences, True
        
        # try to schedule for completion from running queue
        while self.running:
            seq = self.running.popleft()
            # check if we can append one token for current sequence
            if not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if (current_scheduled_tokens >= self.max_num_batched_tokens or 
                    len(scheduled_sequences) >= self.max_num_sequences):
                    break
                self.block_manager.append(seq)   # append one token for sequence
                scheduled_sequences.append(seq)  # add sequence to scheduled list
                current_scheduled_tokens += 1    # update token count
        # re-add to running queue in the same order
        if scheduled_sequences: 
            self.running.extendleft(reversed(scheduled_sequences))

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        '''Preempt a running seq and move it back to waiting queue'''
        self.block_manager.deallocate(seq)  # deallocate blocks for sequence
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        # add back to waiting queue


    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        '''Check whether sequences are finished after generation'''
        for seq, token_id in zip(seqs, token_ids):
            # append token for each sequence
            seq.append_token(token_id)
            # Check stopping conditions:
            #  - EOS token
            #  - Reached max_tokens limit (num of completion tokens)
            #  - Reached max_model_length limit (include prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = 1 + seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length
            # If any stopping condition met, mark as finished and deallocate blocks
            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
