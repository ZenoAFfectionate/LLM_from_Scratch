import os
import json

import numpy as np
from bpe_tokenizer import Tokenizer
from tokenizers import normalizers, Regex


class CompressedTokenizer:
    """
    Compressed Tokenizer implementing vocabulary projection for semantic density.

    This tokenizer compresses the vocabulary space by mapping semantically equivalent
    tokens to canonical IDs using NFKC normalization and lowercasing. This implements
    the projection function P: V → V' as described in the paper, which collapses raw
    token IDs into canonical identifiers based on normalized textual equivalence.

    For a token at position t, we map its raw ID x_t to a canonical ID x'_t = P(x_t)
    to form suffix N-grams g_{t,n} = (x'_{t-n+1}, ..., x'_t).
    """

    def __init__(self, tokenizer_path: str, special_tokens: list[str] = None):
        # initialize the BPE tokenizer from vocabulary and merge files
        self.tokenizer = Tokenizer.from_files(
            vocab_filepath=os.path.join(tokenizer_path, "vocab.json"),
            merges_filepath=os.path.join(tokenizer_path, "merges.txt"),
            special_tokens=special_tokens
        )

        # sentinel token to preserve standalone spaces during normalization
        SENTINEL = "\uE000"

        # text normalization pipeline for canonical form conversion:
        # this implements the projection function P: V → V' by normalizing text
        # to collapse semantically equivalent tokens (e.g., "Apple" vs. " apple")
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),  # Unicode compatibility 
            normalizers.NFD(),   # Canonical decomposition
            normalizers.StripAccents(),  # Remove diacritical marks
            normalizers.Lowercase(),     # Case-insensitive matching
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL), 
            normalizers.Strip(),                 # remove leading/trailing whitespace
            normalizers.Replace(SENTINEL, " "),  # restore standalone spaces
        ])

        # build lookup table mapping raw token IDs to canonical IDs
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        """
        Constructs the lookup table for O(1) token mapping.

        Iterates through the entire original vocabulary, normalizes every
        token, and assigns new unique IDs to unique normalized strings.
        """
        old2new = {}  # map original ID to new canonical ID
        key2new = {}  # map normalized string to new canonical ID
        new_tokens = []

        vocab_size = len(self.tokenizer.decoder_vocab)

        # iterate over every ID in the original vocabulary
        for tid in range(vocab_size):
            # decode token ID back to string representation
            text = self.tokenizer.decode([tid])

            # handle the replacement character:
            if "�" in text:
                key = self.tokenizer.decoder_vocab[tid].decode('utf-8', errors='replace')
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            # check if we have seen this normalized string before
            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)

            old2new[tid] = nid  # map old ID to new ID

        # create a NumPy array for fast vectorized lookup
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        """
        Transforms a sequence of raw input IDs into canonical IDs.
        Uses NumPy for high-performance vectorized mapping.
        """
        arr = np.asarray(input_ids, dtype=np.int64)
        # create a mask for valid token IDs
        pos_mask = arr >= 0
        out = arr.copy()
        # select valid IDs and map using lookup table
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out
    
    def __call__(self, input_ids):
        return self._compress(input_ids)



if __name__ == "__main__":

    datasets = [
        "/home/kemove/Courses/STF_LLM/Assignment_1/data/TinyStories",
        "/home/kemove/Courses/STF_LLM/Assignment_1/data/OpenWebText"
    ]

    for dataset_path in datasets:
        print(f"\nProcessing dataset: {dataset_path}")

        vocab_filepath = os.path.join(dataset_path, "vocab.json")
        merges_filepath = os.path.join(dataset_path, "merges.txt")

        if not os.path.exists(vocab_filepath):
            print(f"  Warning: {vocab_filepath} not found, skipping...")
            continue
        if not os.path.exists(merges_filepath):
            print(f"  Warning: {merges_filepath} not found, skipping...")
            continue

        # Create compressed tokenizer
        print(f"  Loading vocabulary from {vocab_filepath}")
        compressed_tokenizer = CompressedTokenizer(
            vocab_filepath=vocab_filepath,
            merges_filepath=merges_filepath
        )

        # Get compression statistics
        original_vocab_size = len(compressed_tokenizer.tokenizer.decoder_vocab)
        compressed_vocab_size = len(compressed_tokenizer)
        compression_ratio = (1 - compressed_vocab_size / original_vocab_size) * 100

        print(f"  Original vocabulary size: {original_vocab_size}")
        print(f"  Compressed vocabulary size: {compressed_vocab_size}")
        print(f"  Compression ratio: {compression_ratio:.2f}%")

        # Build compressed vocabulary mapping
        # Map old token ID -> new token ID
        compressed_vocab = {
            "lookup_table": compressed_tokenizer.lookup_table.tolist(),
            "original_vocab_size": original_vocab_size,
            "compressed_vocab_size": compressed_vocab_size,
            "compression_ratio": f"{compression_ratio:.2f}%"
        }

        # Save compressed vocabulary to file
        output_filepath = os.path.join(dataset_path, "compressed_vocab.json")
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(compressed_vocab, f, ensure_ascii=False, indent=2)

        print(f"  Compressed vocabulary saved to: {output_filepath}")

    print("\nVocabulary compression completed successfully!")
