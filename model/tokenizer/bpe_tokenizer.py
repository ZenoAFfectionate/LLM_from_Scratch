import os
import time
import json
import regex as re
from tqdm import tqdm
import multiprocessing

from collections import defaultdict
from multiprocessing import Pool
from typing import Iterable, Iterator
from typing import BinaryIO, Dict, List, Tuple


PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> List[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as bytes"

    # total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // max(desired_num_chunks, 1)

    # initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # read-ahead size

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # start at boundary guess

        while True:
            mini_chunk = file.read(mini_chunk_size)  # read a mini chunk

            # if EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    # ensure uniqueness and ordering
    return sorted(set(chunk_boundaries))


def count_word_freqs(text_chunk: str) -> Dict[Tuple[bytes, ...], int]:
    """Count word frequencies in a text chunk (keys are tuples of byte tokens)."""

    def _word2bytes(word: str) -> Tuple[bytes, ...]:
        """Convert a word (string) to a tuple of its byte values (as single-byte bytes)."""
        return tuple(bytes([i]) for i in word.encode("utf-8"))

    word_cnt: Dict[Tuple[bytes, ...], int] = defaultdict(int)

    # find all tokens in the text chunk using the GPT-2 pre-tokenizer regex
    for match in PAT.finditer(text_chunk):
        word_bytes = _word2bytes(match.group(0))
        # only count tokens with at least 2 bytes
        if len(word_bytes) >= 2:
            word_cnt[word_bytes] += 1

    return word_cnt


def process_chunk(
    input_path: str, start: int, end: int, special_tokens: List[str]
) -> Dict[Tuple[bytes, ...], int]:
    """
    Worker: reads a large chunk, splits it by special tokens to respect document
    boundaries, then counts word frequencies.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        text_chunk = chunk_bytes.decode("utf-8", errors="ignore")

    total_word_cnt: Dict[Tuple[bytes, ...], int] = defaultdict(int)

    # build a regex to split by any special token (no stray spaces)
    special_pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"

    # split the large chunk into smaller sub-chunks based on document boundaries
    sub_chunks = re.split(special_pattern, text_chunk)
    special_tokens_set = set(special_tokens)

    for sub_chunk in sub_chunks:
        # skip empty and the special tokens themselves
        if sub_chunk and sub_chunk not in special_tokens_set:
            # get word counts for this sub-chunk
            word_counts_for_sub_chunk = count_word_freqs(sub_chunk)
            # merge into the total word count
            for word, count in word_counts_for_sub_chunk.items():
                total_word_cnt[word] += count

    return total_word_cnt


# --------------------------------------------
# Problem 1: Implement BPE Training Function
# --------------------------------------------
def train_bpe(
    input_path: str, vocab_size: int, special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Train a BPE vocab and merges on the given input data.

    Args:
        input_path: Path to the input text file.
        vocab_size: Desired vocabulary size.
        special_tokens: List of special tokens to include in the vocabulary.

    Returns:
        vocab: Mapping from token id to token bytes.
        merges: List of Byte-Pair Encoding merges (each as a pair of byte-strings).
    """
    assert vocab_size >= 256 + len(special_tokens), "Vocab size is too small!"

    # =========================================
    # step 1: vocabulary and merges initialize
    # =========================================
    print("  > Initializing vocabulary and merges ...", end=" ")
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, token_str in enumerate(special_tokens):
        vocab[256 + i] = token_str.encode("utf-8")

    merges: List[Tuple[bytes, bytes]] = []
    base_vocab_size = len(vocab)
    print("Success!")

    # ======================================
    # step 2: parallelized pre-tokenization
    # ======================================
    print("  > Starting pre-tokenization and word frequency counting ...")

    num_workers = multiprocessing.cpu_count()
    word_dicts: List[Dict[Tuple[bytes, ...], int]] = []
        
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")
        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start < end:  # skip empty chunks
                chunk_args.append((input_path, start, end, special_tokens))

        if chunk_args:
            with Pool(processes=len(chunk_args)) as pool:
                word_dicts = pool.starmap(process_chunk, chunk_args)


    # merge word frequency counts from all chunks
    word_cnt: Dict[Tuple[bytes, ...], int] = defaultdict(int)
    for d in word_dicts:
        for k, v in d.items():
            word_cnt[k] += v

    # =============================================================
    # step 3: optimized merging operation with incremental updates
    # =============================================================
    print("  > Starting BPE merging operation ...")

    # build initial pair frequencies and maintain reverse index
    pair_cnt: Dict[Tuple[bytes, bytes], int] = defaultdict(int) 
    pair_to_words: Dict[Tuple[bytes, bytes], set] = defaultdict(set)
    for word_bytes, cnt in word_cnt.items():
        for pair in zip(word_bytes[:-1], word_bytes[1:]):
            pair_cnt[pair] += cnt
            pair_to_words[pair].add(word_bytes)

    pbar = tqdm(total=vocab_size - base_vocab_size, desc="BPE Merges")

    for i in range(vocab_size - base_vocab_size):
        if not pair_cnt: break  # no more pairs to merge

        # find the most frequent pair, use pair key to break ties deterministically
        max_pair = max(pair_cnt.items(), key=lambda x: (x[1], x[0]))[0]

        # new token id and value
        token_id = base_vocab_size + i
        token_value = max_pair[0] + max_pair[1]
        # update vocab and merges
        vocab[token_id] = token_value
        merges.append(max_pair)

        # Get words that contain the pair to merge from the index
        affected_words = pair_to_words[max_pair]
        new_word_cnt: Dict[Tuple[bytes, ...], int] = {}
        
        updated_pairs = set()  # pairs that need their counts updated
        
        # Process all words in a single pass (sorted for determinism)
        for word_bytes in sorted(word_cnt.keys()):
            cnt = word_cnt[word_bytes]
            # only process words that contain the pair
            if word_bytes in affected_words:
                # Remove old pairs from this word
                for j in range(len(word_bytes) - 1):
                    pair = (word_bytes[j], word_bytes[j + 1])
                    pair_cnt[pair] -= cnt
                    pair_to_words[pair].discard(word_bytes)
                    if pair_cnt[pair] == 0:
                        del pair_cnt[pair]
                        del pair_to_words[pair]
                    updated_pairs.add(pair)

                # create new word with merged token
                new_word_list: List[bytes] = []
                j = 0
                while j < len(word_bytes):
                    if j < len(word_bytes) - 1 and (word_bytes[j], word_bytes[j + 1]) == max_pair:
                        new_word_list.append(token_value)
                        j += 2
                    else:
                        new_word_list.append(word_bytes[j])
                        j += 1
                new_word = tuple(new_word_list)
                new_word_cnt[new_word] = new_word_cnt.get(new_word, 0) + cnt

                # add new pairs from this word
                for j in range(len(new_word) - 1):
                    pair = (new_word[j], new_word[j + 1])
                    if pair not in pair_cnt:
                        pair_cnt[pair] = 0
                        pair_to_words[pair] = set()
                    pair_cnt[pair] += cnt
                    pair_to_words[pair].add(new_word)
                    updated_pairs.add(pair)
            else:
                # copy unaffected word as-is
                new_word_cnt[word_bytes] = cnt

        # update for next iteration
        word_cnt = new_word_cnt
        pbar.update(1)

    pbar.close()
    return vocab, merges


# ------------------------------------------
#  Problem 2: Implement BPE Tokenizer Class
# ------------------------------------------
class Tokenizer:
    """A class to initialize BPE tokenizer and perform Encoding and Decoding."""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Initializes the BPE tokenizer trainer.

        Args:
            vocab_size (dict[int, bytes]): The vocabulary mapping from token IDs to byte sequences.
            merges (list[tuple[bytes, bytes]]): The list of BPE merges.
            special_tokens (list[str] | None): List of special tokens to include in the vocabulary.
        """
        # create encoder and decoder vocabularies
        self.decoder_vocab = vocab
        self.encoder_vocab = {b: i for i, b in vocab.items()}
        # merges is used to indicate the priority of merging pairs
        self.merges = {pair: i for i, pair in enumerate(merges)}

        # handle special tokens
        self.special_tokens_vocab = {}
        self.inverse_special_tokens_vocab = {}

        if special_tokens:
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            # add special tokens to the vocabulary
            next_id = max(self.decoder_vocab.keys()) + 1
            for token_str in sorted_special_tokens:
                token_bytes = token_str.encode("utf-8")
                # add to vocabulary when first encountered
                if token_bytes not in self.encoder_vocab:
                    self.encoder_vocab[token_bytes] = next_id
                    self.decoder_vocab[next_id] = token_bytes
                    next_id += 1
                # store special tokens for separate handling during encoding
                token_id = self.encoder_vocab[token_bytes]
                self.special_tokens_vocab[token_str] = token_id
                self.inverse_special_tokens_vocab[token_id] = token_str

        # Create a regex pattern to find and split by special tokens during encoding
        if self.special_tokens_vocab:
            escaped_tokens = [re.escape(t) for t in self.special_tokens_vocab.keys()]
            self.special_token_pattern = re.compile(f"({'|'.join(escaped_tokens)})")
        else:
            self.special_token_pattern = None

        self.bpe_cache = {}  # Cache for memoizing the result of BPE merging on a word


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        '''Constructs and return a Tokenizer from a serialized vocabulary and list of merges'''
        import json
        # ---------------------------------------------------------------
        # Load vocabulary from JSON file and initialize vocab dictionary
        # ---------------------------------------------------------------
        vocab = {}
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            str_vocab = json.load(f)
        vocab = {val: key.encode('latin-1') for key, val in str_vocab.items()}
        
        # ------------------------------------------------------
        # Load merges from text file and initialize merges list
        # ------------------------------------------------------
        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:  # Only process lines with exactly 2 tokens
                    p1, p2 = parts
                    merges.append((p1.encode('utf-8'), p2.encode('utf-8')))
        
        return cls(vocab, merges, special_tokens)


    def _bpe_merge(self, word_bytes: bytes) -> list[int]:
        ''' Applies the BPE merge rules to a single pre-token (word) '''
        # check cache first for speed
        if word_bytes in self.bpe_cache:
            return self.bpe_cache[word_bytes]
        
        word = [bytes([b]) for b in word_bytes]  # list of single-byte bytes

        if len(word) == 1:  # single byte, no merges possible
            res_ids = [self.encoder_vocab[b''.join(word)]]
            self.bpe_cache[word_bytes] = res_ids
            return res_ids

        # In-place merging loop (fast)
        while True:
            best_i = -1
            best_rank = float('inf')
            # find best adjacent pair
            for i in range(len(word) - 1):
                rank = self.merges.get((word[i], word[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i < 0: break
            # merge at best_i in place
            word[best_i:best_i + 2] = [word[best_i] + word[best_i + 1]]

        # convert the final merged tokens into IDs
        ids = []
        for token in word:
            # look up the token ID, fall back to individual bytes if not found
            if token in self.encoder_vocab:
                ids.append(self.encoder_vocab[token])
            else:
                ids.extend(self.encoder_vocab[bytes([byte])] for byte in token)

        self.bpe_cache[word_bytes] = ids  # save to cache
        return ids


    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        token_ids = []
        
        # If there are no special tokens, we can process the whole text in one go
        if self.special_token_pattern is None:
            # Use re.finditer to iterate over matches instead of re.findall
            for pre_token_match in re.finditer(PAT, text):
                pre_token_bytes = pre_token_match.group(0).encode('utf-8')
                token_ids.extend(self._bpe_merge(pre_token_bytes))
            return token_ids

        # Use finditer to process the text separated by special tokens.
        start_idx = 0
        for match in self.special_token_pattern.finditer(text):
            pre_special_chunk = text[start_idx:match.start()]
            if pre_special_chunk:
                for pre_token_match in re.finditer(PAT, pre_special_chunk):
                    pre_token_bytes = pre_token_match.group(0).encode('utf-8')
                    token_ids.extend(self._bpe_merge(pre_token_bytes))
            
            special_token_str = match.group(0)
            token_ids.append(self.special_tokens_vocab[special_token_str])
            
            start_idx = match.end()

        # handle any remaining text after the last special token
        remaining_chunk = text[start_idx:]
        if remaining_chunk:
            for pre_token_match in re.finditer(PAT, remaining_chunk):
                pre_token_bytes = pre_token_match.group(0).encode('utf-8')
                token_ids.extend(self._bpe_merge(pre_token_bytes))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''Given an iterable of strings, return a generator that lazily yields token IDs'''
        for text_chunk in iterable: yield from self.encode(text_chunk)


    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into a string"""
        byte_chunks = [self.decoder_vocab.get(i, b'') for i in ids]
        # decode the byte sequence to utf-8 text string
        return b''.join(byte_chunks).decode('utf-8', errors='replace')


def save_tokenizer_files(name: str, vocab: dict, merges: list):
    """Helper function to save vocabulary and merges to disk."""
    vocab_path  = os.path.join(DATA_DIR, name, f"vocab.json")
    merges_path = os.path.join(DATA_DIR, name, f"merges.txt")

    # Save vocabulary:
    # The vocab from train_bpe is {id: bytes}. For JSON, we need string keys.
    # We decode the bytes using 'latin-1' which is a safe, lossless.
    inverted_vocab = {v.decode('latin-1'): k for k, v in vocab.items()}
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(inverted_vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to {vocab_path}")

    # Save merges:
    # The merges from train_bpe are list[tuple[bytes, bytes]].
    # We decode them to strings for the text file.
    with open(merges_path, 'w', encoding='utf-8') as f:
        for p1, p2 in merges:
            p1_str = p1.decode('latin-1')
            p2_str = p2.decode('latin-1')
            f.write(f"{p1_str} {p2_str}\n")
    print(f"Merges saved to {merges_path}")


# ------------------------------------------------------------
#  Train BPE Tokenizer on TinyStories and OpenWebText dataset
# ------------------------------------------------------------
if __name__ == "__main__":
    # Define the base directory for data
    DATA_DIR = "/home/kemove/Courses/STF_LLM/Assignment_1/data"
    os.makedirs(DATA_DIR, exist_ok=True)

    datasets = {
        "TinyStories": {
            "input_file": "ts2_train.txt",
            "vocab_size": 10000,
            "special_tokens": ["<|endoftext|>"]
        },
        "OpenWebText": {
            "input_file": "owt_train.txt",
            "vocab_size": 32000,
            "special_tokens": ["<|endoftext|>"]
        }
    }

    # --- Loop through datasets and train tokenizers ---
    for name, config in datasets.items():
        print("-" * 80)
        print(f"🚀 Starting BPE training for '{name}' dataset: ")
        
        input_path = os.path.join(DATA_DIR, name, config["input_file"])
        if not os.path.exists(input_path):
            print(f"⚠️  Warning: Input file not found at {input_path}. Skipping training for '{name}'.")
            continue
            
        start_time = time.time()
        
        # Train the BPE tokenizer
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=config["vocab_size"],
            special_tokens=config["special_tokens"]
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Training for '{name}' completed in {duration:.2f} seconds.")
        
        # Save the trained vocabulary and merges
        save_tokenizer_files(name, vocab, merges)
        print("-" * 80 + "\n")
