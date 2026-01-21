import math
import torch

import numpy as np
import torch.nn as nn
from typing import List

from sympy import isprime

from model.utils import ShortConv
from model.tokenizer.cps_tokenizer import CompressedTokenizer


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    """
    Handles the conversion of raw token sequences into N-gram embedding indices.

    It performs:
    1. Token Compression (Raw IDs -> Canonical IDs)
    2. N-gram Aggregation (Creating windows of history)
    3. Multi-Head Hashing (Mapping N-grams to table indices)
    """
    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_path,
        pad_id,
        seed,  
    ):
        self.vocab_size_per_ngram = engram_vocab_size  # vocab size per n-gram order
        self.max_ngram_size = max_ngram_size           # maximum n-gram order
        self.n_embed_per_ngram = n_embed_per_ngram  # embedding dimension per n-gram order
        self.n_head_per_ngram = n_head_per_ngram    # number of hash head per n-gram order
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        # initalize compressed tokenizer, which map raw tokens to canonical tokens
        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_path=tokenizer_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        # ensure pad_id is also converted to canonical id form
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])
        
        # generate deterministic odd multipliers for hash function.
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        # create unique odd multipliers for each layer
        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            # create unique seed for each layer
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            # generate random integer to serve as coefficients
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1  # ensure odd
            self.layer_multipliers[layer_id] = multipliers
        # calculate the modulo size for every head in n-gram for each layer
        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        """
        Computes distinct prime numbers for each hash head. And using 
        distinct primes reduces the probability that a collision in 
        one head corresponds to a collision in another head.
        """
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            # iterate through n-grams orders
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                # get the base target size for this n-gram
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                # for each head, find the next prime number
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                # record all head sizes for this n-gram
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        """
        Core Hashing Logic: Maps a sequence of token IDs to hash indices.
        """
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            ''' create shifted views of the input for N-gram creation '''
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        # precompute all shifted tokens of the input sequence 
        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        # loop through n-gram orders
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            # extract relevant history for order n
            tokens = base_shifts[:n]
            # implement multiplicative-xor hash
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            # map the large 'mix' integer to the specific table size for each head
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        # stack all hashes into one tensor
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        """
        Public API for compress input IDs and compute hashes for all layers
        """
        # apply semantic compression
        input_ids = self.compressed_tokenizer(input_ids)
        # compute hashes for all layers
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers


class MultiHeadEmbedding(nn.Module):
    """
    This class implements multiple independent embedding tables efficiently by packing them 
    into a single, large physical `nn.Embedding` layer.
    
    Instead of using a `nn.ModuleList` of K separate embedding layers (which would require K 
    separate kernel launches), we use one giant table and manage access via index offsetting.
    
    This is crucial for the "Multi-Head Hashing" mechanism in Engram, ensuring high 
    GPU throughput (O(1) lookup).
    """
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        # calculate prefix sum of the sizes to get offsets and register
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        # calculate the total size and initalize a single embedding table
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # add offsets for head i to every token in the batch
        shifted_input_ids = input_ids + self.offsets
        # perform a single efficient lookup operation
        return self.embedding(shifted_input_ids)


class Engram(nn.Module):
    """
    Engram Module, which implements the Conditional Memory mechanism.
    It orchestrates hashing, embedding lookup, and context-aware gating fusion.
    """
    def __init__(self, 
                 layer_id,     # index of current layer
                 layer_ids,    # list of layer IDs
                 hidden_size,  # model's hidden dimension
                 kernel_size,  # kernel size for shortconv
                 hc_mult,      # hyper-connection multiplicity
                 vocab_size,   # vocab size per n-gram order
                 ngram_size,   # n-gram order
                 embd_dim_per_ngram,  # embedding dimension per n-gram
                 head_num_per_ngram,  # number of heads per n-gram
                 tokenizer_path,      # path to tokenizer
                 pad_id,
                 seed
                ):
        super().__init__()
        self.layer_id = layer_id
        self.hc_mult = hc_mult
        self.hidden_size = hidden_size
        # create n-gram hash mapping
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=vocab_size,
            max_ngram_size = ngram_size,
            n_embed_per_ngram = embd_dim_per_ngram,
            n_head_per_ngram  = head_num_per_ngram,
            layer_ids = layer_ids,
            tokenizer_path=tokenizer_path,
            pad_id = pad_id,
            seed = seed,
        )
        # initialize multi-head embedding
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D = embd_dim_per_ngram // head_num_per_ngram,
        )
        # initialize short convolution
        self.short_conv = ShortConv(
            hidden_size = hidden_size,
            kernel_size = kernel_size,
            dilation    = ngram_size,
            hc_mult     = hc_mult,
        )
        # calculate hidden size and initalize projection layers:
        engram_hidden_size = (ngram_size-1) * embd_dim_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, hidden_size) for _ in range(hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
    
    def forward(self, hidden_states, input_ids):
        # retrieve hash indices for current layer and convert to tensor
        hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])
        # lookup embeddings and flatten across n-gram heads
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        
        # compute gates for each hyper-connection
        gates = []
        for hc_idx in range(self.hc_mult):
            # project and normalize retrieved memory
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            # extract and normalize backbone states
            query = hidden_states[:,:,hc_idx,:]
            normed_query = self.norm2[hc_idx](query)
            # compute gate score
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_size)
            # activation with stabilized sqrt-sigmoid
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        
        gates = torch.stack(gates,dim=2)
        # apply gates to retrieved value projections
        value = gates * self.value_proj(embeddings).unsqueeze(2)
        # perform short convolution and return
        return value + self.short_conv(value)
