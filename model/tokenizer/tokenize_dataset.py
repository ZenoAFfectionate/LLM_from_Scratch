import os
import argparse
from pathlib import Path
import numpy as np
from typing import Iterator
from tqdm import tqdm

from bpe_tokenizer import Tokenizer


# Dataset configurations
DATASETS = {
    "TinyStories": {
        "data_dir": "TinyStories",
        "train_file": "ts2_train.txt",
        "valid_file": "ts2_valid.txt",
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "special_tokens": ["<|endoftext|>"],
    },
    "OpenWebText": {
        "data_dir": "OpenWebText",
        "train_file": "owt_train.txt",
        "valid_file": "owt_valid.txt",
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "special_tokens": ["<|endoftext|>"],
    },
}


def read_text_chunks(file_path: str, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """Generator that yields text chunks for memory-efficient processing."""
    file_size = os.path.getsize(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading") as pbar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                pbar.update(len(chunk.encode('utf-8')))
                yield chunk


def tokenize_and_save_data(
    text_file: str,
    tokenizer: Tokenizer,
    output_file: str,
    chunk_size: int = 1024 * 1024,
    force: bool = False
):
    """
    Memory-efficient tokenization using encode_iterable.
    
    Args:
        text_file: Path to input text file
        tokenizer: BPE tokenizer instance
        output_file: Path to output binary file
        chunk_size: Size of text chunks to read at once
        force: If True, overwrite existing output file
    """
    if os.path.exists(output_file) and not force:
        print(f"  Tokenized data already exists: {output_file}")
        print(f"  Use --force to overwrite.")
        return

    print(f"  Tokenizing: {text_file}")
    file_size = os.path.getsize(text_file)
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")

    # Use encode_iterable for memory-efficient tokenization
    text_chunks = read_text_chunks(text_file, chunk_size)

    # Process tokens in batches to avoid memory issues
    token_batch_size = 10_000_000  # Process 10M tokens at a time
    token_buffer = []
    total_tokens = 0

    with open(output_file, 'wb') as out_f:
        for token_id in tokenizer.encode_iterable(text_chunks):
            token_buffer.append(token_id)

            # Write batch when buffer is full
            if len(token_buffer) >= token_batch_size:
                token_array = np.array(token_buffer, dtype=np.int32)
                token_array.tofile(out_f)
                total_tokens += len(token_buffer)
                print(f"  Processed {total_tokens:,} tokens...")
                token_buffer = []

        # Write remaining tokens
        if token_buffer:
            token_array = np.array(token_buffer, dtype=np.int32)
            token_array.tofile(out_f)
            total_tokens += len(token_buffer)

    output_size = os.path.getsize(output_file)
    print(f"  Saved {total_tokens:,} tokens to {output_file}")
    print(f"  Output size: {output_size:,} bytes ({output_size / 1024 / 1024:.1f} MB)")
    print(f"  Compression ratio: {file_size / output_size:.2f}x")


def load_tokenizer(data_dir: Path, config: dict) -> Tokenizer:
    """Load tokenizer from vocab and merges files."""
    vocab_path = data_dir / config["vocab_file"]
    merges_path = data_dir / config["merges_file"]
    special_tokens = config["special_tokens"]

    print(f"  Loading tokenizer from {vocab_path}")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=special_tokens
    )
    print(f"  Vocabulary size: {len(tokenizer.decoder_vocab):,}")
    return tokenizer


def process_dataset(dataset_name: str, base_dir: Path, force: bool = False):
    """Process a single dataset."""
    if dataset_name not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return

    config = DATASETS[dataset_name]
    data_dir = base_dir / config["data_dir"]

    print("=" * 60)
    print(f"Processing dataset: {dataset_name}")
    print("=" * 60)

    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    # Load tokenizer
    tokenizer = load_tokenizer(data_dir, config)

    # Define file paths
    train_file = data_dir / config["train_file"]
    valid_file = data_dir / config["valid_file"]
    train_bin = data_dir / "tokens_train.bin"
    valid_bin = data_dir / "tokens_valid.bin"

    # Process training data
    if train_file.exists():
        print(f"\n[Train Data]")
        tokenize_and_save_data(str(train_file), tokenizer, str(train_bin), force=force)
    else:
        print(f"  Warning: Train file not found: {train_file}")

    # Process validation data
    if valid_file.exists():
        print(f"\n[Valid Data]")
        tokenize_and_save_data(str(valid_file), tokenizer, str(valid_bin), force=force)
    else:
        print(f"  Warning: Valid file not found: {valid_file}")

    print()


def verify_tokenized_data(data_dir: Path, tokenizer: Tokenizer, num_tokens: int = 100):
    """Verify tokenized data by decoding a sample."""
    train_bin = data_dir / "tokens_train.bin"
    if not train_bin.exists():
        return

    print(f"  Verifying tokenized data...")
    tokens = np.fromfile(str(train_bin), dtype=np.int32)[:num_tokens]
    decoded = tokenizer.decode(tokens.tolist())
    print(f"  First {num_tokens} tokens decode to:")
    print(f"  '{decoded[:200]}...'")


def main():
    parser = argparse.ArgumentParser(description='Tokenize text datasets for language model training')
    
    parser.add_argument('--dataset', type=str, default='all', choices=['TinyStories', 'OpenWebText', 'all'])
    parser.add_argument('--data_dir', type=str, default=None, help='Base data directory')
    parser.add_argument('--force', action='store_true', help='Overwrite existing tokenized files')
    parser.add_argument('--verify', action='store_true', help='Verify tokenized data by decoding')
    args = parser.parse_args()

    # Determine base data directory
    if args.data_dir:
        base_dir = Path(args.data_dir)
    else:
        # Default: data directory relative to this script
        script_dir = Path(__file__).parent
        base_dir = script_dir.parent.parent / "data"

    base_dir = base_dir.resolve()
    print(f"Data directory: {base_dir}")

    if not base_dir.exists():
        print(f"Error: Data directory not found: {base_dir}")
        return

    # Process datasets
    if args.dataset == 'all':
        for dataset_name in DATASETS:
            process_dataset(dataset_name, base_dir, force=args.force)
    else:
        process_dataset(args.dataset, base_dir, force=args.force)

    # Verify if requested
    if args.verify:
        print("=" * 60)
        print("Verification")
        print("=" * 60)
        for dataset_name in (DATASETS.keys() if args.dataset == 'all' else [args.dataset]):
            config = DATASETS[dataset_name]
            data_dir = base_dir / config["data_dir"]
            if data_dir.exists():
                print(f"\n[{dataset_name}]")
                tokenizer = load_tokenizer(data_dir, config)
                verify_tokenized_data(data_dir, tokenizer)

    print("Done!")


if __name__ == "__main__":
    main()
