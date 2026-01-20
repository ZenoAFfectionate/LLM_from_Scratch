from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    Basic unit of memory in vLLM: stores a fixed-size chunk of tokens
    Manages reference count (4 block reuse) and hash (4 cache lookup)
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    Core manager for VLLM's block-based memory system:
    - Allocates/deallocates blocks to sequences
    - Reuses blocks via hash-based caching (reduces memory duplication)
    - Manages free/used block pools with reference counting
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # initalize all blocks sequentially from 0 to num_blocks-1
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash map: block hash -> block ID (for fast cache lookup)
        self.hash_to_block_id: dict[int, int] = dict()
        # record free_block ids and used_block ids
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        Compute a unique hash for a block of tokens (with optional prefix for continuity)
        - prefix: hash of the previous block (ensures unique hash for sequential blocks)
        - Uses xxh64 (fast, non-cryptographic hash) for performance
        """
        h = xxhash.xxh64()
        # add prefix to bytes stream (little-endian, 8 bytes)
        if prefix != -1: h.update(prefix.to_bytes(8, "little"))
        # convert token list to numpy array then to bytes
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        # block must be free before allocation:
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)  # removing from free queue
        self.used_block_ids.add(block_id)     # adding to used set
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)  # removing from used set
        self.free_block_ids.append(block_id)  # adding to free queue

    def can_allocate(self, seq: Sequence) -> bool:
        '''  Check if enough free blocks are available for allocation '''
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table, "Sequence already has allocated blocks"
        h = -1
        cache_miss = False
        # iterate over each block needed by the sequence
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  # get token id for i-th block
            # compute hash only if block is full (match block size)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            # check cache: get block id from hash map
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            # Handle cache miss: allocate new block from free pool
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            # Handle cache hit: reuse existing block
            else:
                seq.num_cached_tokens += self.block_size
                # already in used blocks: increment count
                if block_id in self.used_block_ids:
                    self.blocks[block_id].ref_count += 1
                # not in used blocks: allocate and use it
                else:
                    block = self._allocate_block(block_id)
            # update block hash/tokens and cache map
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        # deallocate in reverse order to consistent with allocation order
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # reset sequence's block state
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # case 1: next token needs a new block
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1, ""
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # case 2: last block is now full
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1, ""
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            # compute hash and update block/cache
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        # case 3: last block is partially filled
        else:
            assert last_block.hash == -1
