import math
import torch
import pickle
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from model.transformer import TransformerLM
from model.config import Config
from engine.sequence import Sequence
from utils.context import set_context, get_context, reset_context
from utils.sampler import Sampler


# =========================================================
# ModelRunner adapted for TransformerLM with Paged Attention
# =========================================================


class ModelRunner:
    """
    ModelRunner for running the TransformerLM model with paged attention.
    
    Key adaptations:
    - Uses TransformerLM instead of Qwen3ForCausalLM
    - Maps Config parameters to engine config format
    - Allocates paged KV cache and assigns to attention layers
    """
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        self.event = event

        # set distributed config
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        self.enforce_eager = config.get('enforce_eager', False)

        self.rank = rank
        dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)
        torch.cuda.set_device(rank)

        # Build model config from engine config
        model_config = Config(
            vocab_size=config['vocab_size'],
            context_length=config['max_model_length'],
            d_model=config['hidden_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            num_kv_heads=config.get('num_kv_heads', config['num_heads']),
            d_ff=config.get('intermediate_size', config['hidden_size'] * 4),
            dropout=0.0,  # No dropout during inference
            attention_type=config.get('attention_type', 'GQA'),
            rope_theta=config.get('rope_theta', 10000.0),
            rope_dim=config.get('rope_dim', None),
            q_lora_rank=config.get('q_lora_rank', None),
            kv_lora_rank=config.get('kv_lora_rank', None),
            use_moe=config.get('use_moe', False),
            n_routed_experts=config.get('n_routed_experts', 8),
            num_experts_per_tok=config.get('num_experts_per_tok', 1),
            n_shared_experts=config.get('n_shared_experts', 1),
        )
        
        # Create model with paged attention enabled
        self.model = TransformerLM(
            config=model_config,
            device=f'cuda:{rank}',
            dtype=torch.bfloat16,
        ).cuda(rank)
        
        # Enable paged attention mode on all attention layers
        self._enable_paged_attention()
        
        self.sampler = Sampler()

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = torch.bfloat16  # Use bfloat16 for KV cache

        # warm up model so that we know peak memory usage
        self.warmup_model()
        # allocate kv cache
        self.allocate_kv_cache()
        # capture cuda graph for decoding
        if not self.enforce_eager:
            self.capture_cudagraph()
    
        torch.set_default_device(f'cuda:{self.rank}')
        torch.set_default_dtype(self.default_dtype)

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__
    
    def _enable_paged_attention(self):
        """Enable paged attention mode on all attention layers."""
        for layer in self.model.layers:
            if hasattr(layer, 'att'):
                # Set paged attention flag
                layer.att.paged_attention = True
                # Set block size for paged attention
                layer.att.block_size = self.block_size
                # Disable the old per-sequence cache
                layer.att.cache_enabled = False

    # only use read when rank != 0
    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # Check if process group exists before destroying
        if dist.is_initialized():
            dist.destroy_process_group()
    
    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args) # Unpack args when calling
            if method_name == 'exit':
                self.exit()
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: dict):
        if self.world_size > 1 and self.rank == 0: # will be called in main engine
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length
        seqs = [Sequence(token_ids=[0]*max_model_length) for _ in range(batch_size)]
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # find all available memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem * self.config['gpu_memory_utilization']
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        # reserve some room for peak memory usage during model execution
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # find parameters to compute kv cache size
        num_layers = self.config['num_layers']
        num_kv_heads = self.config.get('num_kv_heads', self.config['num_heads']) // self.world_size
        head_dim = self.config.get('head_dim', self.config['hidden_size'] // self.config['num_heads'])
        
        # For MLA attention, the KV cache stores compressed representations
        attention_type = self.config.get('attention_type', 'GQA')
        if attention_type == 'MLA':
            # MLA caches kv_lora_rank + rope_dim instead of 2 * num_kv_heads * head_dim
            kv_lora_rank = self.config.get('kv_lora_rank', self.config['hidden_size'] // 4)
            rope_dim = self.config.get('rope_dim', 16)
            cache_dim = kv_lora_rank + rope_dim  # compressed KV + position encoding
            # MLA uses different cache structure: (kv_compressed, pe)
            block_bytes = self.block_size * num_layers * cache_dim * self.default_dtype.itemsize
        else:
            # GQA/MHA: standard K and V caches
            # block_bytes = block_size * 2(K+V) * num_layers * num_kv_heads * head_dim * dtype_size
            block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        
        self.num_available_kv_blocks = int(available_mem // block_bytes)
        assert self.num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        
        print(f"[Rank {self.rank}] Allocated {self.num_available_kv_blocks} KV cache blocks "
              f"(block_size={self.block_size}, attention_type={attention_type})")

        # allocate max possible kv cache for the model
        # this is the key for paged attention: one giant KV cache pool, divided into blocks
        if attention_type == 'MLA':
            # MLA: separate kv_cache and pe_cache
            # Shape: (num_layers, num_blocks, block_size, cache_dim)
            kv_cache = torch.empty(
                num_layers, self.num_available_kv_blocks, self.block_size, kv_lora_rank,
                device=f'cuda:{self.rank}', dtype=self.default_dtype
            )
            pe_cache = torch.empty(
                num_layers, self.num_available_kv_blocks, self.block_size, rope_dim,
                device=f'cuda:{self.rank}', dtype=self.default_dtype
            )
            layer_id = 0
            for layer in self.model.layers:
                if hasattr(layer, 'att'):
                    layer.att.k_cache = kv_cache[layer_id]  # kv compressed
                    layer.att.v_cache = pe_cache[layer_id]  # position encoding
                    layer_id += 1
        else:
            # GQA/MHA: standard K and V caches
            # Shape: (2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
            allocated_kv_cache = torch.empty(
                2, num_layers, self.num_available_kv_blocks, self.block_size, num_kv_heads, head_dim,
                device=f'cuda:{self.rank}', dtype=self.default_dtype
            )
            layer_id = 0
            for layer in self.model.layers:
                if hasattr(layer, 'att'):
                    layer.att.k_cache = allocated_kv_cache[0, layer_id]
                    layer.att.v_cache = allocated_kv_cache[1, layer_id]
                    layer_id += 1

    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache into consideration: 
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # length: sum of all input_ids after prefix cache
        input_ids = []
        # length: sum of all input_ids after prefix cache
        slot_mappings = []
        # length: num_seqs
        seqlens_q = []
        # length: num_seqs
        seqlens_k = []
        # length: num_seqs + 1
        cu_seqlens_q = [0]
        # length: num_seqs + 1
        cu_seqlens_k = [0]
        # block_tables: num_seqs x num_blocks (padded)
        block_tables = []
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    else:
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # pad block_tables
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
                block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids


    # prepare input data for decoding
    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []   
        slot_mappings = []  
        block_tables = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)
        # TransformerLM expects (batch_size, seq_len), so reshape to (batch_size, 1)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True).unsqueeze(1)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids    

    # prepare the temperature
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill or self.enforce_eager:
            # For prefill with paged attention:
            # - input_ids shape depends on implementation
            # TransformerLM.forward expects (batch, seq_len)
            # For varlen prefill, we use batch=1 and concatenate all tokens
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)  # (1, total_tokens)
            logits = self.model(input_ids)
        else:
            # Decode phase with CUDA graph
            bs = input_ids.size(0)
            context = get_context()

            # finds smallest captured graph that fits the batch size
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars
            # copy input data into graph variables
            # input_ids shape: (batch_size, 1) for decode
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'][:bs].fill_(-1)
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars["block_tables"].zero_()
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # replay the graph
            graph.replay()
            logits = vars['outputs'][:bs].clone()

        return logits


    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)
        
        # Handle different logits shapes
        # TransformerLM returns (batch, seq_len, vocab_size)
        if logits.dim() == 3:
            # For prefill: get last token logits for each sequence
            if is_prefill:
                context = get_context()
                # Extract last token logits for each sequence based on cu_seqlens_q
                cu_seqlens = context.cu_seqlens_q
                last_positions = cu_seqlens[1:] - 1  # positions of last tokens
                logits = logits.squeeze(0)[last_positions]  # (num_seqs, vocab_size)
            else:
                # For decode: squeeze the seq_len dimension
                logits = logits.squeeze(1)  # (batch, vocab_size)
        
        # only sample when rank == 0
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))
        reset_context()
        return token_ids

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated once and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = self.config['max_num_seqs']
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)
        
        # for decoding, input is always of shape (batch_size, 1)
        # Note: TransformerLM expects (batch_size, seq_len), so we use (batch_size, 1) for decode
        input_ids = torch.zeros(max_bs, 1, dtype=torch.long, device=f'cuda:{self.rank}')
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # how many tokens each sequence has processed
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # where to read KV values in the cache
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        # output logits - TransformerLM returns (batch, seq_len, vocab_size)
        outputs = torch.zeros(max_bs, 1, self.config['vocab_size'], device=f'cuda:{self.rank}', dtype=self.default_dtype)

        # graphs to be captured for different batch sizes
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )
            # Warmup run before capture
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    graph_pool = graph.pool()
            # store the captured graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
