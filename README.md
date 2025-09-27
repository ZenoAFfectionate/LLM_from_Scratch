# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Detailed Running Instructions

Based on the project structure with `data/`, `config/`, and `checkpoints/` directories, here are the detailed instructions for running the complete pipeline:

### Project Structure Overview
```
/home/kemove/Courses/STF_LLM/Assignment_1/
├── data/
│   ├── TinyStories/       # TinyStories dataset with tokenized files
│   └── OpenWebText/       # OpenWebText dataset
├── config/                # Configuration files
├── checkpoints/           # Saved model checkpoints
├── cs336_basics/          # Core implementation modules
├── train.py              # Training script
└── generate.py           # Text generation script
```

### 1. Building BPE Tokenizer (bpe_tokenizer.py)

The BPE tokenizer is automatically configured to process both datasets:

```bash
cd /home/kemove/Courses/STF_LLM/Assignment_1
python cs336_basics/bpe_tokenizer.py
```

This will:
- Build BPE tokenizer for **TinyStories** dataset (vocab_size: 10,000)
- Build BPE tokenizer for **OpenWebText** dataset (vocab_size: 32,000)
- Save vocabulary and merges files to respective data directories

**Output files:**
- `data/TinyStories/vocab.json` & `data/TinyStories/merges.txt`
- `data/OpenWebText/vocab.json` & `data/OpenWebText/merges.txt`

### 2. Training the Model (train.py)

#### TinyStories Training:
```bash
python train.py --config config/train_tinystories.json
```
- Model Architecture: 8 layers, 16 heads, 512 d_model, 1344 d_ff
- Context length: 1024 tokens
- Batch size: 16
- Max iterations: 10,000
- Learning rate: 3e-4 (max) → 3e-5 (min) with cosine schedule

Resume Training from Checkpoint:
```bash
python train.py --config config/train_tinystories.json --resume checkpoints/TinyStories/best_model.pt
```

#### OpenWebText Training:
```bash
python train.py --config config/train_openwebtext.json
```

- Model Architecture: 12 layers, 16 heads, 768 d_model, 3072 d_ff
- Context length: 2048 tokens
- Batch size: 8
- Max iterations: 50,000
- Learning rate: 1e-4 (max) → 1e-5 (min) with cosine schedule

Resume Training from Checkpoint:
```bash
python train.py --config config/train_openwebtext.json --resume checkpoints/OpenWebText/best_model.pt
```


### 3. Text Generation (generate.py)

#### Custom Prompt Generation:
```bash
python generate.py --config config/generate_tinystories.json --prompt "Once upon a time"
```

#### Interactive Mode:
```bash
python generate.py --config config/generate_tinystories.json
```

#### Override Generation Parameters:
```bash
python generate.py \
    --config config/generate_tinystories.json \
    --prompt "The little girl" \
    --max_new_tokens 150 \
    --temperature 0.7 \
    --top_p 0.95
```

#### Use Specific Checkpoint:
```bash
python generate.py \
    --config config/generate_tinystories.json \
    --checkpoint checkpoints/TinyStories/final_model.pt \
    --prompt "In a magical forest"
```

### 4. Complete Workflow Example

For a full end-to-end run:

```bash
# 1. Build tokenizers for both datasets
python cs336_basics/bpe_tokenizer.py

# 2. Train model on TinyStories
python train.py --config config/train_tinystories.json

# 3. Generate text using trained model
python generate.py --config config/generate_tinystories.json --prompt "Once upon a time, there was a brave little mouse"
```

### Key Configuration Files:
- `config/train_tinystories.json` - Training configuration for TinyStories
- `config/train_openwebtext.json` - Training configuration for OpenWebText
- `config/generate_tinystories.json` - Generation configuration for TinyStories
- `config/generate_openwebtext.json` - Generation configuration for OpenWebText

Your checkpoints are saved in `checkpoints/TinyStories/` or `checkpoints/OpenWebText/` with `best_model.pt` and `final_model.pt` available for generation.

