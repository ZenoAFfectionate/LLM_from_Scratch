import os
from pathlib import Path
import numpy as np
from typing import Iterator

from bpe_tokenizer import Tokenizer


def read_text_chunks(file_path: str, chunk_size: int = 1024*1024) -> Iterator[str]:
    """Generator that yields text chunks for memory-efficient processing"""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            yield chunk


def tokenize_and_save_data(text_file: str, tokenizer: Tokenizer, output_file: str, chunk_size: int = 1024*1024):
    """Memory-efficient tokenization using encode_iterable"""
    if os.path.exists(output_file):
        print(f"Tokenized data already exists: {output_file}")
        return

    print(f"Tokenizing {text_file} using memory-efficient streaming...")

    # Get file size for progress tracking
    file_size = os.path.getsize(text_file)
    print(f"File size: {file_size:,} bytes")

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
                print(f"Processed {total_tokens:,} tokens...")
                token_buffer = []

        # Write remaining tokens
        if token_buffer:
            token_array = np.array(token_buffer, dtype=np.int32)
            token_array.tofile(out_f)
            total_tokens += len(token_buffer)

    print(f"Saved {total_tokens:,} tokens to {output_file}")


if __name__ == "__main__":
    """  """
    

    data_dir = Path("")

    # 
    train_file = data_dir / "train.txt"
    valid_file = data_dir / "valid.txt"
    # Paths for text and tokenized data
    train_bin = data_dir / f"tokens_train.bin"  # tokenized train data
    valid_bin = data_dir / f"tokens_valid.bin"  # tokenized valid data

    # Process training data
    print("Using memory-efficient tokenization for train data...")
    tokenize_and_save_data(str(train_file), tokenizer, str(train_bin))

    # Process validation data
    print("Using memory-efficient tokenization for valid data...")
    tokenize_and_save_data(str(valid_file), tokenizer, str(valid_bin))
