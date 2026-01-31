import xxhash
import numpy as np
from collections import deque

from sequence import Sequence


class Block:
    """
    Basic unit of memory in vLLM: stores a fixed-size chunk of tokens
    Manages reference count (4 block reuse) and hash (4 cache lookup)
    """
    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []

    def update(self, h: int, token_ids: list[int]):
        self.hash = h 
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []


class BlockManager:
    """
    Core manager for VLLM's block-based memory system:
    - Allocates/deallocates blocks to sequences
    - Reuses blocks via hash-based caching (reduces memory duplication)
    - Manages free/used block pools with reference counting
    """
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size: int = block_size  # num of token per block
        # initalize all blocks sequentially from 0 to num_blocks-1
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash map: block hash -> block ID (for fast cache lookup)
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        """
        Compute a unique hash for a block of tokens (optional prefix for continuity)
        - prefix: hash of the previous block (unique hash for sequence continuity)
        - Uses xxh64 (fast, non-cryptographic hash) for performance
        """
        h = xxhash.xxh64()
        # add prefix to bytes stream (little-endian, 8 bytes)
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        # convert token list to numpy array then to bytes
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()


    def _allocate_block(self, block_id: int) -> Block:
        """Allocate a block and add it to the used list"""
        block = self.blocks[block_id]
        # check if the block is already allocated
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()  # reset block state
        # remove from free list and add to used list
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        """Deallocate a block and add it to the free list"""
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        block.token_ids = []  # reset token IDs
        # remove from used list and add to free list
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)


    def can_allocate(self, seq: Sequence) -> bool:
        '''Check if we can allocate a block for this sequence'''
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence) -> None:
        """Allocate blocks for this sequence"""
        h = -1
        for i in range(seq.num_blocks):
            no_cache_found = False
            token_ids = seq.block(i)
            # only compute hash for full blocks, always -1 for partial blocks
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # if cache miss or hash collision
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                seq.num_cached_tokens += self.block_size
                # allocate a new block if not used yet
                if block_id not in self.used_block_ids:
                    block = self._allocate_block(block_id)
                # update block information if already used
                else:
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
            else:
                # allocate a new block and update its information
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                if h != -1: self.hash_to_block_id[h] = block.block_id
            
            seq.block_table.append(block.block_id)  # update block table
    
    def deallocate(self, seq: Sequence) -> None:
        """Deallocate blocks for this sequence"""
        # update block information
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            # deallocate if no longer used
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # update sequence information
        seq.block_table = []
        seq.num_cached_tokens = 0


    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append tokens to this sequence"""
        if seq.num_tokens % self.block_size == 0:
            return len(self.free_block_ids) > 0
        return True

    def append(self, seq: Sequence) -> None:
        '''Append tokens to this sequence'''
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]

        # if the last block is now full, compute hash
        if seq.num_tokens % self.block_size == 0:
            prefix = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash
            h = self.compute_hash(token_ids = seq.block(seq.num_blocks - 1), prefix_hash_value=prefix)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self.hash_to_block_id[h] = block.block_id
        # if one new block is needed
        elif seq.num_tokens % self.block_size == 1:
            # Previous block should be finalized
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])
            block_tables.append(block.block_id)
        # else, do nothing
        else:
            assert last_block_for_seq_id in self.used_block_ids,  "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"
