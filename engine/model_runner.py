import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from model.config import Config
from engine.sequence import Sequence
from model.transformer import TransformerLM
from utils.sampler import Sampler
from utils.context import set_context, get_context, reset_context
from utils.loader import load_model


class ModelRunner:
    """
    Model runner for inference with TransformerLM.

    This class handles:
    - Model initialization and weight loading
    - KV cache allocation and management
    - Prefill and decode phases
    - Multi-GPU coordination (tensor parallelism)
    - CUDA graph capturing for efficient decoding
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.block_size = getattr(config, 'kvcache_block_size', 256)
        self.enforce_eager = getattr(config, 'enforce_eager', True)
        self.world_size = getattr(config, 'tensor_parallel_size', 1)
        self.rank = rank
        self.event = event

        # Initialize NCCL process group for inter-GPU communication
        if self.world_size > 1:
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # Specify default dtype and device for model
        default_dtype = torch.get_default_dtype()
        model_dtype = getattr(config, 'torch_dtype', torch.float32)
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype, torch.float32)
        torch.set_default_dtype(model_dtype)
        torch.set_default_device("cuda")

        # Initialize model
        self.model = TransformerLM(config=config, device="cuda", dtype=model_dtype)

        # Load model weights
        model_path = getattr(config, 'checkpoint_path', None) or getattr(config, 'model', None)
        if model_path:
            load_model(self.model, model_path)

        self.sampler = Sampler()

        # Phase-based setup
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # Shared memory setup for multi-GPU coordination
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if self.world_size > 1:
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = getattr(self.config, 'max_num_batched_tokens', 16384)
        max_model_len = getattr(self.config, 'max_model_len', self.config.context_length)
        max_num_seqs = getattr(self.config, 'max_num_seqs', 512)
        num_seqs = min(max_num_batched_tokens // max_model_len, max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # Get model dimensions from config
        num_layers = config.num_layers
        num_kv_heads = getattr(config, 'num_kv_heads', config.num_heads) // self.world_size
        head_dim = config.d_model // config.num_heads

        # Get dtype size
        model_dtype = getattr(config, 'torch_dtype', torch.float32)
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype, torch.float32)
        dtype_size = torch.tensor([], dtype=model_dtype).element_size()

        block_bytes = 2 * num_layers * self.block_size * num_kv_heads * head_dim * dtype_size
        gpu_memory_utilization = getattr(config, 'gpu_memory_utilization', 0.9)
        num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes

        if num_kvcache_blocks <= 0:
            print("Warning: Not enough memory for KV cache, using minimal allocation")
            num_kvcache_blocks = 1

        # Store for later use
        self.num_kvcache_blocks = num_kvcache_blocks

        # Allocate KV cache tensor
        self.kv_cache = torch.empty(
            2, num_layers, num_kvcache_blocks, self.block_size, num_kv_heads, head_dim,
            device="cuda", dtype=model_dtype
        )

        # Assign KV cache to attention layers
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        print(f"Allocated KV cache: {num_kvcache_blocks} blocks, {layer_id} layers")

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))

        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        Run the model forward pass.

        Note: TransformerLM returns logits directly, unlike Qwen3ForCausalLM which
        has separate forward() and compute_logits() methods.
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # For TransformerLM: reshape to (batch, seq_len) if needed
            if input_ids.dim() == 1:
                # For prefill: use the context to determine sequence boundaries
                context = get_context()
                if context.cu_seqlens_q is not None and len(context.cu_seqlens_q) > 1:
                    # Variable length sequences - process each sequence
                    logits_list = []
                    cu_seqlens = context.cu_seqlens_q.tolist()
                    for i in range(len(cu_seqlens) - 1):
                        start, end = cu_seqlens[i], cu_seqlens[i + 1]
                        seq_input = input_ids[start:end].unsqueeze(0)
                        seq_logits = self.model(seq_input)
                        # Get last token logits for each sequence
                        logits_list.append(seq_logits[:, -1, :])
                    return torch.cat(logits_list, dim=0)
                else:
                    # Single sequence or batch with same length
                    input_ids = input_ids.unsqueeze(0)
                    logits = self.model(input_ids)
                    return logits[:, -1, :]  # Last token logits
            else:
                logits = self.model(input_ids)
                return logits[:, -1, :]
        else:
            # CUDA graph path for decode
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return graph_vars["outputs"][:bs]

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """Capture CUDA graphs for efficient decode phase."""
        max_num_seqs = getattr(self.config, 'max_num_seqs', 512)
        max_model_len = getattr(self.config, 'max_model_len', self.config.context_length)

        max_bs = min(max_num_seqs, 512)
        max_num_blocks = (max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        positions = torch.zeros(max_bs, dtype=torch.int64, device="cuda")
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device="cuda")
        outputs = torch.zeros(max_bs, self.config.vocab_size, device="cuda")

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])

            # Warmup - TransformerLM expects (batch, seq_len), decode uses (batch, 1)
            batch_input = input_ids[:bs].unsqueeze(1)  # Shape: (bs, 1)
            warmup_output = self.model(batch_input)
            outputs[:bs] = warmup_output[:, -1, :]

            with torch.cuda.graph(graph, self.graph_pool):
                batch_input = input_ids[:bs].unsqueeze(1)
                graph_output = self.model(batch_input)
                outputs[:bs] = graph_output[:, -1, :]

            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
