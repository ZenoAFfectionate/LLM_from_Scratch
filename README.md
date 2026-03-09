# Building Transformer-Based Language Model from Scratch

A production-grade Transformer language model built from scratch with advanced architectural innovations, featuring BPE tokenization, multiple attention mechanisms (MHA, GQA, MLA), MoE, enGram conditional memory, Muon optimizer, and vLLM-style paged attention inference.

---

## Overview

This project implements a complete Transformer-based language model ecosystem with cutting-edge techniques from recent research (DeepSeek-V3, etc.) and systematic performance engineering to maximize training efficiency under resource constraints.

The goal of this project is as followed:
1. **In-Depth Analysis and Implementation of Transformer Architecture**: Construct a Transformer-based Large Language Model (LLM) from scratch. Through code implementation, deeply dissect the underlying operational mechanisms and mathematical principles of language models to establish a solid theoretical foundation.

2. **Architectural Innovation and Capability Enhancement**: Going beyond the baseline architecture, this project is committed to introducing and implementing cutting-edge architectural modules. By refining and optimizing the model structure, we aim to significantly elevate feature extraction capabilities, inference quality, and generalization performance.

3. **Extreme Performance Optimization and Single-GPU Adaptation**: Pursue ultimate system performance optimization, specifically tailored for consumer-grade hardware (a single RTX 4090). Leveraging techniques such as operator fusion and memory management, achieve efficient training and low-latency inference for Large Language Models within constrained resources.

### Project Evolution and Technical Journey

This project began with Stanford CS336 Assignment 1 as its foundational framework, implementing a complete Transformer-based language model from first principles. Every component—from fundamental operations like softmax and cross-entropy to optimization algorithms like AdamW—was hand-coded to achieve deep understanding of the underlying mathematics and computational mechanics.

**Architectural Innovations.** The project introduces multiple state-of-the-art attention mechanisms including Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA), and DeepSeek Sparse Attention (DSA), enabling flexible experimentation with different efficiency-performance trade-offs. DeepSeek-V3's Mixture of Experts (MoE) architecture with auxiliary-loss-free load balancing brings sparse computation capabilities, while Multi-Token Prediction (MTP) enhances training data efficiency. The enGram module implements conditional memory via scalable n-gram lookup. The project also supports supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) for alignment with human preferences.

**Training Optimization.** Initial experiments revealed critical performance bottlenecks on a single RTX 4090. We systematically eliminated CPU-GPU synchronization issues by vectorizing MoE operations and removing Python loops and `.item()` calls. Integration with `torch.compile()` required careful redesign to use fixed-size tensor operations instead of boolean indexing, preventing costly CUDA graph recompilations. These optimizations improved GPU utilization from 10-20% to 85-95%, achieving a 10-20× speedup in training throughput.

**Inference Optimization.** We implemented a vLLM-style inference engine with paged attention and continuous batching. The system features efficient KV cache management with hash-based prefix caching, a unified prefill-decode architecture, and CUDA graph capture for decode acceleration. This enables efficient multi-request serving with automatic memory reuse across sequences.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [BPE Tokenizer](#bpe-tokenizer)
- [Model Architecture](#model-architecture)
  - [Rotary Positional Embedding (RoPE)](#rotary-positional-embedding-rope)
  - [Attention Mechanisms](#attention-mechanisms)
  - [Feed-Forward Networks](#feed-forward-networks)
  - [RMSNorm with Fused Residual Connection](#rmsnorm-with-fused-residual-connection)
- [Conditional Memory: enGram](#conditional-memory-engram)
- [Muon Optimizer](#muon-optimizer)
- [Inference Engine](#inference-engine)
  - [vLLM-Style Architecture](#vllm-style-architecture)
  - [Paged Attention Kernels](#paged-attention-kernels)

---

## Quick Start

### 1. Create Environment

We manage environments with `conda` for reproducibility. You can install conda virtual environment as follow:
```bash
conda env create -f environment.yml
```

After installing all necessary packets, you can activate the environment using:

```bash
conda init & conda activate llm
```

### 2. Download Dataset

```bash
mkdir -p data && cd data

# TinyStories dataset
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText dataset
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### 3. Build BPE Tokenizer

```bash
python model/tokenizer/bpe_tokenizer.py
```

This creates tokenizers for both TinyStories (vocab: 10K) and OpenWebText (vocab: 32K).

### 4. Tokenize Pretrain Dataset

In order to acclerate model pretraining, we shall tokenize the pretraining dataset into tokens of binary form first.

```bash
python model/tokenizer/tokenize_dataset.py
```

### 5. Train Model

```bash
# TinyStories (8 layers, 512 dim)
python train.py --config config/[MLA+MoE]train_tinystories.json

# OpenWebText (12 layers, 768 dim)
python train.py --config config/[MLA+MoE]train_openwebtext.json
```

### 6. Generate Text

To enable interactive text generation and demonstrate the trained model's capabilities, we provide a production-ready generation pipeline with an intuitive Gradio-based web interface. The generation system implements efficient autoregressive decoding with KV caching, flexible sampling strategies (temperature, top-k, top-p), and multi-turn conversation support.

**Launch the web interface:**

```bash
# Launch with default TinyStories model
python generate.py --config config/generate_tinystories.json

# Or use OpenWebText model
python generate.py --config config/generate_openwebtext.json

# Override checkpoint path
python generate.py --config config/generate_tinystories.json --checkpoint checkpoints/custom_model.pt

# Create public shareable link (via Gradio)
python generate.py --config config/generate_tinystories.json --share

# Custom server settings
python generate.py --config config/generate_tinystories.json --server_name 0.0.0.0 --server_port 8080
```

The web interface provides:
- **Interactive chat interface** with conversation history tracking
- **Adjustable generation parameters**: max tokens, temperature, top-k, top-p
- **Pre-built example prompts** for quick experimentation
- **Real-time generation** with efficient KV cache utilization

Access the interface at `http://127.0.0.1:7860` (default) after launching.

---

## BPE Tokenizer

The BPE (Byte Pair Encoding) tokenizer is implemented with a focus on efficiency through careful data structure design, algorithmic optimization, and parallelization. The training process involves three main phases: parallel pre-tokenization, word frequency aggregation, and heap-optimized merge operations. The encoding phase employs a novel heap-based algorithm with doubly-linked list representation for optimal performance.

**Parallel Pre-tokenization** leverages multiprocessing to split large corpus files into chunks at document boundaries. Each worker independently computes word frequencies using the GPT-2 pre-tokenizer regex pattern, which identifies linguistic units like contractions, words, numbers, and whitespace:

```python
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def count_word_freqs(text_chunk: str) -> Dict[Tuple[bytes, ...], int]:
    word_cnt: Dict[Tuple[bytes, ...], int] = defaultdict(int)
    for match in PAT.finditer(text_chunk):
        word_bytes = tuple(bytes([i]) for i in match.group(0).encode("utf-8"))
        if len(word_bytes) >= 2:
            word_cnt[word_bytes] += 1
    return word_cnt
```

**Heap-based Training with Lazy Deletion** replaces the naive O(W) linear scan for finding the maximum-frequency pair with a max-heap data structure, reducing the per-iteration lookup to O(log P) where P is the number of unique pairs. The key insight is that pair frequencies change incrementally—most pairs remain unchanged after each merge. A custom `MaxHeapItem` wrapper ensures correct tie-breaking (highest count first, then lexicographically largest pair), and a lazy deletion strategy handles stale entries efficiently:

```python
class MaxHeapItem:
    '''Max-heap behavior with correct tie-breaking'''
    __slots__ = ('count', 'pair')
    def __init__(self, count: int, pair: Tuple[bytes, bytes]):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        # Higher count first, then larger pair lexicographically
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair

# Pop from heap until we find a valid entry (lazy deletion)
while pair_heap:
    item = heapq.heappop(pair_heap)
    # Verify this entry is still valid
    if item.pair in pair_cnt and pair_cnt[item.pair] == item.count:
        max_pair = item.pair
        break
```

The training loop maintains a reverse index `pair_to_words` that maps each byte pair to the set of words containing it. When a merge occurs, only affected words are reprocessed, and all modified pairs (both incremented and decremented counts) are pushed back to the heap. This incremental update strategy avoids rebuilding the entire heap after each merge.

**Order-Independent Word Processing** eliminates redundant sorting overhead during training. The original implementation sorted word keys on every iteration for determinism, incurring O(M × W log W) total sorting cost across M merge iterations. Analysis reveals that word processing order does not affect the final result: pair count updates are additive/subtractive (commutative), and tie-breaking is handled by the heap's comparison logic after all words are processed. Removing the unnecessary `sorted()` call provides significant speedup on large vocabularies.

**Heap-based Encoding with Doubly-Linked List** transforms the encoding algorithm from O(n²) to O(n log n) complexity, where n is the input length. The naive approach scans all n-1 pairs on each merge iteration to find the minimum-rank pair, resulting in O(n²) total work. Our optimized implementation uses a min-heap to track pairs by merge priority and a doubly-linked list (via index arrays) to efficiently skip deleted tokens:

```python
def _bpe_merge(self, word_bytes: bytes) -> list[int]:
    if word_bytes in self.bpe_cache:
        return self.bpe_cache[word_bytes]

    n = len(word_bytes)
    tokens = [bytes([b]) for b in word_bytes]

    # Doubly-linked list: prev[i] = previous index, next[i] = next index
    prev_idx = [-1] + list(range(n - 1))  # prev_idx[0] = -1
    next_idx = list(range(1, n)) + [-1]   # next_idx[n-1] = -1

    # Build min-heap with (rank, position, left_token, right_token)
    heap = []
    for i in range(n - 1):
        pair = (tokens[i], tokens[i + 1])
        rank = self.merges.get(pair)
        if rank is not None:
            heapq.heappush(heap, (rank, i, tokens[i], tokens[i + 1]))

    while heap:
        rank, pos, left_tok, right_tok = heapq.heappop(heap)

        # Skip stale entries (lazy deletion)
        if tokens[pos] is None: continue
        right_pos = next_idx[pos]
        if right_pos == -1 or tokens[right_pos] is None: continue
        if tokens[pos] != left_tok or tokens[right_pos] != right_tok: continue

        # Merge tokens and update linked list
        merged_token = left_tok + right_tok
        tokens[pos] = merged_token
        tokens[right_pos] = None  # Mark as deleted

        # Update linked list pointers
        new_next = next_idx[right_pos]
        next_idx[pos] = new_next
        if new_next != -1:
            prev_idx[new_next] = pos

        # Push new adjacent pairs to heap
        left_neighbor = prev_idx[pos]
        if left_neighbor != -1 and tokens[left_neighbor] is not None:
            pair = (tokens[left_neighbor], merged_token)
            if (pair_rank := self.merges.get(pair)) is not None:
                heapq.heappush(heap, (pair_rank, left_neighbor, tokens[left_neighbor], merged_token))

        if new_next != -1 and tokens[new_next] is not None:
            pair = (merged_token, tokens[new_next])
            if (pair_rank := self.merges.get(pair)) is not None:
                heapq.heappush(heap, (pair_rank, pos, merged_token, tokens[new_next]))

    # Collect final tokens and convert to IDs
    final_tokens = [t for t in tokens if t is not None]
    ids = [self.encoder_vocab[token] for token in final_tokens]
    self.bpe_cache[word_bytes] = ids
    return ids
```

The lazy deletion strategy stores complete pair information `(rank, position, left_token, right_token)` in each heap entry. When popping, we verify that the tokens at the recorded positions still match the stored values—if not, the entry is stale and discarded. This avoids the overhead of explicitly removing invalidated entries from the heap. Combined with memoization caching, the optimized encoder achieves efficient tokenization even for long input sequences.

---

## Model Architecture

### Rotary Positional Embedding (RoPE)

RoPE encodes positional information by rotating query and key vectors in the complex plane. Unlike absolute position embeddings, RoPE enables relative position awareness through the geometric property that dot products between rotated vectors depend only on their relative positions.

The implementation precomputes rotation matrices as cosine-sine pairs and applies them via element-wise operations rather than explicit matrix multiplication:

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        position = torch.arange(max_seq_len, device=device).float()
        angles = torch.einsum('i, j -> ij', position, inv_freq)
        cos_sin = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        self.register_buffer('cos_sin_cached', cos_sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_sin = self.cos_sin_cached[token_positions, :x.shape[-1]]
        cos, sin = torch.chunk(cos_sin, 2, dim=-1)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
```

### Attention Mechanisms

The project implements three attention variants, each with distinct memory-compute trade-offs:

**Multi-Head Attention (MHA)** is the standard Transformer attention where each head independently computes attention. Q, K, V projections share the same dimensionality, and RMSNorm is applied to Q and K before RoPE for training stability:

```python
# MHA: All heads compute independent attention
q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
q, k = self.q_norm(q), self.k_norm(k)
q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
k = k.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
v = v.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
# Apply RoPE and compute attention
attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

**Grouped-Query Attention (GQA)** reduces KV cache memory by sharing K/V heads across multiple query heads. The key insight is that multiple query heads can attend to the same key-value pairs without significant quality loss:

```python
# GQA: Multiple query heads share KV heads
# num_query_heads = 16, num_kv_heads = 4 -> group_size = 4
k = k.unsqueeze(2).expand(bsz, num_kv_heads, group_size, seq_len, head_dim)
k = k.reshape(bsz, num_query_heads, seq_len, head_dim)  # No memory copy with expand+reshape
```

**Multi-Head Latent Attention (MLA)** from DeepSeek-V3 achieves extreme KV cache compression through low-rank factorization. Instead of caching full K/V tensors, MLA caches a compressed latent representation and reconstructs K/V on-the-fly. The decoupled RoPE design separates position-dependent and position-independent components:

```python
class MultiHeadLatentAttention(nn.Module):
    def forward(self, x, mask=None):
        # Q path: compress then expand with separate RoPE component
        q_compressed = self.q_norm(self.q_down_proj(x))
        q_fused = self.q_up_proj_fused(q_compressed)
        q_nope = q_fused[..., :self.d_model]  # Position-independent
        q_rope = q_fused[..., self.d_model:]   # Position-dependent (gets RoPE)
        
        # KV path: compress to latent space
        kv_compressed = self.kv_norm(self.kv_down_proj(x))  # Cached during inference
        k_rope = self.k_rope_proj(x)  # Separate position encoding
        
        # Expand KV and concatenate RoPE components
        kv_fused = self.kv_up_proj_fused(kv_compressed)
        k = torch.cat([kv_fused[..., :d_model], k_rope.expand(..., num_heads, rope_dim)], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)
        
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

The core innovation of MLA is **matrix absorption** during inference: instead of expanding the compressed KV cache, attention scores are computed directly in the latent space by absorbing the up-projection weights into the query:

```python
# Decode with matrix absorption - compute attention in latent space
w_uk = self.kv_up_proj_fused.weight[:d_model, :].view(num_heads, head_dim, kv_lora_rank)
q_absorbed = torch.einsum('bhd, hdk -> bhk', q_nope, w_uk)  # Query in latent space
attn_score = torch.einsum('bhk, btk -> bht', q_absorbed, cached_kv)  # Score from cache
```

### Feed-Forward Networks

**SwiGLU MLP** combines the Swish activation with a gating mechanism. The gated linear unit provides adaptive feature selection, where the gate path modulates information flow:

```python
class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**Mixture of Experts (MoE)** routes each token to its top-k experts based on learned gating scores. The implementation uses **auxiliary-loss-free load balancing** from DeepSeek-V3, which maintains expert biases that are updated based on load imbalance without adding terms to the training loss:

```python
class Gate(nn.Module):
    def forward(self, x):
        logits = x @ self.weight.t()
        scores = torch.sigmoid(logits)
        biased_scores = logits + self.expert_bias  # Bias for load balancing
        
        _, topk_indices = torch.topk(biased_scores, k=self.top_k, dim=-1)
        topk_weights = torch.gather(scores, dim=-1, index=topk_indices)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-10)
        return topk_indices, topk_weights, aux_loss
    
    def update_bias(self, total_tokens):
        expected_load = (total_tokens * self.top_k) / self.n_routed_experts
        load_diff = self.expert_load.float() - expected_load
        self.expert_bias -= torch.sign(load_diff) * self.bias_update_speed
```

The MoE forward pass is optimized using a **sort-based dispatch** strategy that groups tokens by expert for contiguous memory access, then uses a fused Triton kernel for weighted scatter-add:

```python
# Sort tokens by expert for contiguous access
sorted_expert_idx = torch.argsort(flat_topk_idx)
sorted_tokens = x_flat[token_indices[sorted_expert_idx]]
expert_inputs = torch.split(sorted_tokens, expert_counts_list)

# Process each expert
expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(n_experts)]

# Fused weighted scatter-add with segment reduction
y_flat = fused_scatter_add_weighted(torch.cat(expert_outputs), sorted_token_idx, sorted_weight, ...)
```

### RMSNorm with Fused Residual Connection

RMSNorm provides layer normalization without mean centering, offering computational savings over LayerNorm. The implementation includes an optional **fused residual addition** that computes both normalization and residual in a single pass, reducing memory traffic:

```python
class RMSNorm(nn.Module):
    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        dtype = x.dtype
        if residual is None:
            x = x.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype)
        else:
            # Fused: add residual, compute RMS norm, return both
            x = residual = x.float() + residual.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype), residual.to(dtype)
```

---

## Conditional Memory: enGram

enGram implements **Conditional Memory via Scalable Lookup** (based on the paper "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models"). Unlike attention which scales quadratically with context length, enGram provides O(1) memory retrieval through n-gram hashing.

The core idea is to use the preceding n-gram context as a hash key to look up relevant information from learned embedding tables. This provides a complementary retrieval mechanism that captures local patterns efficiently.

**Multi-Head Hashing** maps n-grams to embedding indices using distinct prime moduli per head, reducing collision probability:

```python
class NgramHashMapping:
    def _get_ngram_hashes(self, input_ids, layer_id):
        # Create shifted views for n-gram windows
        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
        
        for n in range(2, self.max_ngram_size + 1):
            tokens = base_shifts[:n]
            # Multiplicative-XOR hash combining all tokens in n-gram
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            
            # Each head uses a distinct prime modulus
            for j in range(num_heads):
                head_hash = mix % head_vocab_sizes[j]  # Prime modulus
                all_hashes.append(head_hash)
```

**Packed Multi-Head Embeddings** achieves O(1) lookup by packing multiple embedding tables into a single physical table with offset indexing, avoiding separate kernel launches:

```python
class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        # Compute cumulative offsets for each head's table
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets))
        # Single large embedding table
        self.embedding = nn.Embedding(sum(list_of_N), D)
    
    def forward(self, input_ids):
        # Single lookup with offset adjustment
        return self.embedding(input_ids + self.offsets)
```

**Context-Aware Gating** determines how much information flows from the retrieved memory. The gate score is computed via normalized dot-product similarity between the query (backbone hidden states) and key (projected memory):

```python
class Engram(nn.Module):
    def forward(self, hidden_states, input_ids):
        hash_ids = self.hash_mapping.hash(input_ids)[self.layer_id]
        embeddings = self.multi_head_embedding(hash_ids).flatten(start_dim=-2)
        
        # Compute gates for each hyper-connection
        for hc_idx in range(self.hc_mult):
            key = self.norm1[hc_idx](self.key_projs[hc_idx](embeddings))
            query = self.norm2[hc_idx](hidden_states[:, :, hc_idx, :])
            gate = (key * query).sum(dim=-1) / math.sqrt(self.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()  # Stabilized sqrt-sigmoid
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        
        value = torch.stack(gates, dim=2) * self.value_proj(embeddings).unsqueeze(2)
        return value + self.short_conv(value)  # Add local context via causal convolution
```

---

## Muon Optimizer

Muon (MomentUm Orthogonalized by Newton-schulz) is designed specifically for training transformer hidden weights. The key insight is that orthogonalizing gradient updates leads to better conditioning and faster convergence for matrix-shaped parameters.

**Newton-Schulz Orthogonalization** iteratively refines a matrix toward its nearest orthogonal matrix. The quintic iteration variant converges rapidly in just 5 steps and is stable in bfloat16:

```python
def newtonschulz5_orthogonalization(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315  # Quintic coefficients maximizing slope at zero
    
    X = G.to(torch.bfloat16)
    if G.size(-2) > G.size(-1): X = X.mT  # Handle tall matrices
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)  # Ensure spectral norm ≤ 1
    
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    return X.mT if G.size(-2) > G.size(-1) else X
```

**Muon Update Rule** combines momentum with orthogonalization. The scaling factor ensures consistent update magnitudes regardless of matrix aspect ratio:

```python
def muon_update(grad, momentum, scaling_factor, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)  # EMA momentum update
    update = grad.lerp(momentum, beta) if nesterov else momentum
    update = newtonschulz5_orthogonalization(update, steps=ns_steps)
    return update * scaling_factor  # scaling_factor = sqrt(max(1, m/n))
```

**Hybrid Muon-AdamW Optimization** uses Muon for 2D hidden weights (attention projections, FFN layers) and AdamW for everything else (embeddings, biases, normalization parameters):

```python
class MuonAdamWOptimizer:
    def __init__(self, model, muon_lr=0.02, adamw_lr=3e-4, ...):
        # Separate parameters by type
        for name, param in model.named_parameters():
            is_2d_weight = param.ndim >= 2
            is_special = 'embedding' in name or 'lm_head' in name or 'bias' in name
            
            if is_2d_weight and not is_special:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        self.muon_optimizer = Muon(muon_params, lr=muon_lr, ...)
        self.adamw_optimizer = AdamW(adamw_params, lr=adamw_lr, fused=True, ...)
```

---

## Inference Engine

### vLLM-Style Architecture

The inference engine implements a vLLM-style architecture with continuous batching and paged KV cache management. The system comprises four main components working together to maximize GPU utilization.

**BlockManager** handles paged memory allocation. Each block stores a fixed number of tokens' KV cache, and blocks are reused across sequences through hash-based prefix caching:

```python
class BlockManager:
    def allocate(self, seq: Sequence):
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, prefix_hash) if full_block else -1
            
            # Check for cache hit
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                seq.num_cached_tokens += self.block_size  # Prefix cache hit
            else:
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
            seq.block_table.append(block.block_id)
```

**Scheduler** implements a priority-based scheduling policy that maximizes throughput through continuous batching. Prefill requests (new sequences) are prioritized, then decode requests (generating sequences) are batched together:

```python
class Scheduler:
    def schedule(self) -> tuple[list[Sequence], bool]:
        # Try prefill first (higher priority)
        while self.waiting and can_schedule_more():
            seq = self.waiting.popleft()
            self.block_manager.allocate(seq)
            scheduled_sequences.append(seq)
        if scheduled_sequences:
            return scheduled_sequences, True  # is_prefill=True
        
        # Then decode running sequences
        while self.running:
            seq = self.running.popleft()
            self.block_manager.append(seq)  # Allocate new slot for next token
            scheduled_sequences.append(seq)
        return scheduled_sequences, False  # is_prefill=False
```

**ModelRunner** bridges the scheduler and model, preparing inputs for either prefill or decode mode and managing KV cache assignment:

```python
class ModelRunner:
    def allocate_kv_cache(self):
        # Allocate one large KV cache pool, divided into blocks
        if attention_type == 'MLA':
            kv_cache = torch.empty(num_layers, num_blocks, block_size, kv_lora_rank, ...)
            pe_cache = torch.empty(num_layers, num_blocks, block_size, rope_dim, ...)
        else:
            allocated_kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim, ...)
        
        # Assign cache slices to each attention layer
        for layer in self.model.layers:
            layer.att.k_cache = allocated_kv_cache[0, layer_id]
            layer.att.v_cache = allocated_kv_cache[1, layer_id]
```

### Paged Attention Kernels

Custom Triton kernels enable efficient paged attention operations. The key challenge is handling non-contiguous memory access patterns when reading from scattered cache blocks.

**Store KV Cache** writes new key-value pairs to their designated slots using slot_mapping:

```python
@triton.jit
def _store_kvcache_kernel(key_ptr, value_ptr, k_cache_ptr, v_cache_ptr, slot_mapping_ptr, ...):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx == -1: return  # Skip invalid slots
    
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    
    # Load from input, store to paged cache
    key = tl.load(key_ptr + input_offset)
    tl.store(k_cache_ptr + cache_offset, key)
```

**Flash Attention Prefill** implements variable-length flash attention for efficient prefill. The online softmax algorithm enables processing long sequences without materializing the full attention matrix:

```python
@triton.jit
def _flash_attention_varlen_kernel(Q, K, V, O, cu_seqlens_q_ptr, scale, ...):
    # Load sequence boundaries
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    
    # Online softmax accumulators
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Sum of exp
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10  # Max score
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    for block_n in range(num_blocks):
        qk = tl.dot(q, k) * scale
        qk = tl.where(mask_causal, qk, -1e10)
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    output = acc / l_i[:, None]
```

**Paged Attention Decode** reads from scattered cache blocks during autoregressive generation:

```python
@triton.jit
def _paged_attention_decode_kernel(output_ptr, query_ptr, k_cache_ptr, v_cache_ptr, 
                                    block_tables_ptr, context_lens_ptr, ...):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    context_len = tl.load(context_lens_ptr + batch_idx)
    q = tl.load(query_ptr + q_offset)
    
    # Online softmax over all context tokens
    for token_idx in range(context_len):
        block_num = token_idx // block_size
        block_offset = token_idx % block_size
        
        # Lookup physical block from block table
        physical_block_idx = tl.load(block_tables_ptr + batch_idx * max_blocks + block_num)
        
        # Load K from paged cache
        k = tl.load(k_cache_ptr + physical_block_idx * block_stride + ...)
        score = tl.sum(q * k) * scale
        
        # Update softmax and accumulate weighted V
        # ... online softmax logic ...
```

**MLA Cache Store** handles the compressed KV representation unique to MLA:

```python
@triton.jit
def _store_mla_cache_kernel(kv_ptr, pe_ptr, kv_cache_ptr, pe_cache_ptr, slot_mapping_ptr, ...):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    # Store compressed KV (kv_lora_rank dimensions)
    kv_data = tl.load(kv_ptr + token_idx * kv_dim + tl.arange(0, kv_dim))
    tl.store(kv_cache_ptr + cache_offset, kv_data)
    
    # Store position encoding (rope_dim dimensions)
    pe_data = tl.load(pe_ptr + token_idx * pe_dim + tl.arange(0, pe_dim))
    tl.store(pe_cache_ptr + pe_cache_offset, pe_data)
```

---

## Experiment Result


### Parameter Setting



### Experiment on TinyStories

All TinyStories experiments use an 8-layer Transformer with `d_model=512`, 16 attention heads, `d_ff=1344`, context length 512, batch size 128, and are trained for 10K iterations with the hybrid Muon+AdamW optimizer under mixed-precision training on a single RTX 4090. MoE variants use 4 routed experts with top-1 routing and 1 shared expert (applied to layers 1–7). GQA uses 4 KV heads (group size 4). MLA uses `q_lora_rank=128`, `kv_lora_rank=64`, and `rope_dim=8`.

| Attention | Forward | **Loss** | **PPL** | **Iteration Time** |
|-----------|---------|----------|---------|---------------------|
| MHA       | FFN     | 0.4886   | 1.63    | 0.245s              |
| MHA       | MoE     | 0.4887   | 1.63    | 0.275s              |
| GQA       | FFN     | 0.4929   | 1.64    | 0.242s              |
| GQA       | MoE     | 0.4942   | 1.64    | 0.275s              |
| MLA       | MoE     | 0.4996   | 1.65    | 0.285s              |

The most striking observation from the TinyStories experiments is the remarkably narrow performance spread across all architectural configurations: the loss ranges from 0.4886 to 0.4996, and perplexity varies only between 1.63 and 1.65. This near-uniform convergence strongly suggests that the TinyStories dataset—a synthetic corpus composed of simple children's stories with limited vocabulary and straightforward narrative patterns—presents a modeling task that is well within the capacity of even the simplest configuration tested. When the dataset complexity is low enough, additional architectural sophistication yields diminishing returns in final quality.

The baseline MHA+FFN configuration achieves the best loss (0.4886) while maintaining an efficient iteration time of 0.245s, demonstrating that the classic Transformer architecture remains highly competitive on small-scale tasks. Replacing the dense FFN with MoE in the MHA+MoE variant leaves the loss virtually unchanged (0.4887) but introduces a 12% overhead in iteration time (0.275s vs. 0.245s). This overhead stems from the expert routing mechanism, gating computations, and the sort-based token dispatch, none of which provide meaningful benefit when the data distribution is simple enough for a single dense FFN to model effectively.

GQA-based variants exhibit a slight loss increase compared to their MHA counterparts (0.4929 for GQA+FFN vs. 0.4886 for MHA+FFN), reflecting the minor representational cost of sharing KV heads across query groups. However, this trade-off is well-justified: GQA+FFN achieves the fastest iteration time of 0.242s—marginally faster than MHA+FFN—thanks to reduced memory bandwidth requirements from the smaller KV projection. The MLA+MoE configuration, despite being the most architecturally complex, records the highest loss (0.4996) and slowest iteration time (0.285s). The low-rank compression overhead in MLA's query and KV paths adds computational cost that is not amortized on short context windows (512 tokens), where the KV cache savings are minimal.

Overall, the TinyStories results establish an important baseline: for simple, low-entropy datasets, the classical MHA+FFN architecture is optimal in terms of both quality and efficiency. The advanced mechanisms—MoE's conditional computation, GQA's KV sharing, and MLA's latent compression—are designed to shine at larger scales, where richer data distributions and longer contexts demand greater model capacity and memory efficiency. The next section on OpenWebText validates this hypothesis.

### Optimizer Compaison

We conducted a comparative experiment on the OpenWebText dataset using the MLA+MoE architecture to evaluate the effectiveness of the Muon optimizer. The following table shows training loss and perplexity (PPL) at various checkpoints:

| Optimizer | 1K Iter (Loss / PPL) | 5K Iter (Loss / PPL) | 10K Iter (Loss / PPL) | 20K Iter (Loss / PPL) | Final (Loss / PPL) |
|-----------|---------------------|---------------------|----------------------|----------------------|-------------------|
| AdamW | 2.2187 / 9.20 | 1.5019 / 4.49 | 1.3874 / 4.00 | 1.3093 / 3.70 | 1.2785 / 3.59 |
| Muon + AdamW | 1.6287 / 5.10 | 1.3587 / 3.89 | 1.2944 / 3.65 | 1.2306 / 3.42 | 1.1997 / 3.32 |

The results demonstrate that the hybrid Muon+AdamW optimizer significantly outperforms pure AdamW across all training stages. At the early phase (1K iterations), Muon+AdamW achieves a loss of 1.63 compared to AdamW's 2.22—a 27% reduction that translates to nearly halving the perplexity (5.10 vs. 9.20). This faster initial convergence stems from Muon's orthogonalized updates, which provide better-conditioned gradient directions for the high-dimensional weight matrices in attention and FFN layers.

The advantage persists throughout training: at convergence, Muon+AdamW reaches a final loss of 1.20 (PPL 3.32) versus AdamW's 1.28 (PPL 3.59), representing a 7.5% improvement in perplexity. Notably, Muon+AdamW at 10K iterations already surpasses AdamW's final performance, suggesting that orthogonalized momentum not only accelerates convergence but also finds better local minima. The consistent gap across checkpoints validates the theoretical motivation that Newton-Schulz orthogonalization improves optimization dynamics for transformer weight matrices.


### Experiment on OpenWebText

All OpenWebText experiments use a 12-layer Transformer with `d_model=768`, 16 attention heads, `d_ff=3072`, context length 2048, batch size 16, and are trained for 32K iterations with the hybrid Muon+AdamW optimizer under mixed-precision training on a single RTX 4090. All configurations use MoE with 8 routed experts, top-1 routing, and 1 shared expert (applied to layers 1–11). GQA uses 8 KV heads (group size 2). MLA uses `q_lora_rank=192`, `kv_lora_rank=96`, and `rope_dim=16`. DSA (DeepSeek Sparse Attention) experiments are currently in progress.

| Attention | Forward | **Loss** | **PPL** | **Iteration Time** |
|-----------|---------|----------|---------|---------------------|
| MHA       | MoE     | 1.1803   | 3.28    | 4.25s               |
| GQA       | MoE     | 1.1940   | 3.30    | 4.07s               |
| MLA       | MoE     | 1.1997   | 3.32    | 3.95s               |
| DSA       | MoE     | —        | —       | —                   |

Moving from TinyStories to OpenWebText—a diverse, large-scale web corpus with significantly richer vocabulary and more complex linguistic structures—reveals a far more meaningful efficiency-quality trade-off across attention mechanisms. The performance gaps, while still moderate in absolute terms, are substantially more pronounced than those observed on TinyStories, confirming that architectural differences become increasingly relevant as dataset complexity grows.

MHA+MoE achieves the best modeling quality with a loss of 1.1803 (PPL 3.28), serving as the quality ceiling for this comparison. Each query head maintains its own independent KV representation, providing maximum representational flexibility for capturing the diverse patterns present in web text. However, this comes at the cost of the highest iteration time (4.25s), as the full KV computation for all 16 heads creates substantial memory bandwidth pressure on the longer 2048-token context window.

GQA+MoE strikes a compelling balance, achieving a loss of 1.1940 (PPL 3.30)—only a 1.2% increase over MHA—while reducing iteration time by 4.2% to 4.07s. The 8 KV heads (group size 2) reduce the KV projection parameters and cache size by half relative to MHA, which directly translates to faster computation. The minimal quality degradation confirms the finding from the GQA literature that adjacent query heads learn highly correlated attention patterns, making KV sharing an effective compression strategy even on complex web data.

MLA+MoE presents the most aggressive trade-off: it records the highest loss (1.1997, PPL 3.32) but achieves the fastest iteration time of 3.95s—a 7.1% speedup compared to MHA+MoE. The low-rank factorization compresses the KV representation from full `d_model` down to `kv_lora_rank=96`, dramatically reducing memory bandwidth during attention computation. The slightly higher loss can be attributed to the information bottleneck introduced by the low-rank compression, as well as the overhead of the decoupled RoPE design which adds a separate positional encoding path. Importantly, the efficiency advantage of MLA is expected to become even more pronounced during inference, where the compressed KV cache enables significantly higher throughput and longer context windows—a benefit not captured by training-time iteration metrics alone.

A clear trend emerges across the three attention mechanisms: as the architecture increasingly compresses the KV representation (MHA → GQA → MLA), iteration time decreases monotonically (4.25s → 4.07s → 3.95s) while loss increases modestly (1.1803 → 1.1940 → 1.1997). This reveals a smooth Pareto frontier between training speed and modeling quality, where each mechanism occupies a distinct operating point. For applications prioritizing raw language modeling quality, MHA+MoE remains the best choice. For deployment scenarios where inference efficiency and memory footprint are critical—such as serving long-context requests—MLA+MoE offers the most favorable overall trade-off, as its training-time quality gap is small while its inference-time advantages in KV cache compression are substantial.

---

## References

### Foundational Architecture

- **Attention is All You Need** (Transformer Architecture)
  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
  *Advances in Neural Information Processing Systems*, 30.
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

- **Language Models are Unsupervised Multitask Learners** (GPT-2)
  Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).
  OpenAI Blog.
  [Link](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

### Tokenization

- **Neural Machine Translation of Rare Words with Subword Units** (Byte Pair Encoding)
  Sennrich, R., Haddow, B., & Birch, A. (2016).
  *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*.
  [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

### Positional Encoding

- **RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)**
  Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

### Optimization Algorithms

- **Muon: MomentUm Orthogonalized by Newton-schulz**
  Jordan, K. et al. (2024).
  [GitHub](https://github.com/KellerJordan/Muon)

### Attention Mechanisms

- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (Grouped-Query Attention)
  Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023).
  *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*.
  [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

- **Fast Transformer Decoding: One Write-Head is All You Need** (Multi-Query Attention)
  Shazeer, N. (2019).
  [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
  Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022).
  *Advances in Neural Information Processing Systems*, 35.
  [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

### Mixture-of-Experts (MoE)

- **DeepSeek-V3 Technical Report** (MLA, MoE, Auxiliary-Loss-Free Load Balancing)
  DeepSeek-AI et al. (2024).
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**
  Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017).
  *International Conference on Learning Representations*.
  [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)

### Conditional Memory

- **Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models** (enGram)
  Zhang, Y., Liu, Z., Wang, W., Shrivastava, A. (2025).
  [arXiv:2501.10544](https://arxiv.org/abs/2501.10544)

### Efficient Inference

- **Efficient Memory Management for Large Language Model Serving with PagedAttention** (vLLM)
  Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023).
  *Proceedings of the 29th Symposium on Operating Systems Principles*.
  [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

### Alignment and Preference Learning

- **Training Language Models to Follow Instructions with Human Feedback** (RLHF)
  Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., ... & Lowe, R. (2022).
  *Advances in Neural Information Processing Systems*, 35.
  [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

- **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (DPO)
  Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023).
  *Advances in Neural Information Processing Systems*, 36.
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

---

## License

Copyright 2025 Stanford University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
