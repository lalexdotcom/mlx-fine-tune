# MLX Fine-Tune

A pipeline for fine-tuning MLX language models on HuggingFace datasets, optimized for Apple Silicon. Features automatic caching at every stage, early stopping, and LoRA adapter fusion.

---

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- [LM Studio](https://lmstudio.ai) (optional, for model management)

No manual dependency installation required — the script handles everything automatically via a dedicated virtual environment.

---

## Installation

Clone or copy the scripts into a directory of your choice:
```
mlx-fine-tune/
  run.sh         ← main entry point
  attach.sh      ← reconnect to a running pipeline
  terminate.sh   ← stop a running pipeline
  pipeline.py    ← pipeline logic
```

Make the scripts executable:
```bash
chmod +x run.sh attach.sh terminate.sh
```

---

## Pipeline Overview

The pipeline runs in sequential stages, each with its own cache layer:
```
HuggingFace dataset (Parquet)
        ↓  [cached per dataset]
    Raw JSONL
        ↓  [cached per dataset + chat template]
  Masked JSONL (train / valid / test)
        ↓  [resumable, early stopping]
   LoRA Adapters
        ↓
   Fused Model
```

**Stage 1 — Parquet download**: fetches `.parquet` files from the HuggingFace Datasets API and caches them locally.

**Stage 2 — Raw JSONL conversion**: converts Parquet row groups to a flat JSONL file using `pyarrow`. Streamed row-group by row-group to keep memory usage low.

**Stage 3 — Chat template application**: applies the model's Jinja2 chat template (read from `tokenizer_config.json`) to produce masked training examples. Only `train_on_turn: true` turns are included in the loss. Cache is keyed on both dataset and chat template — switching to a model with the same template reuses the cache.

**Stage 4 — LoRA fine-tuning**: runs `mlx_lm lora` with early stopping. Validation loss is tracked at each checkpoint; training stops automatically when it has not improved for `--patience` consecutive validations. The best checkpoint is saved to `best_iter.txt` for later use. If a previous run exists for the same model/dataset combination, training resumes automatically from the last checkpoint.

**Stage 5 — Adapter fusion**: merges the best LoRA adapter into the base model using `mlx_lm fuse`, producing a standalone model ready to load in LM Studio.

---

## Usage

### Basic fine-tuning
```bash
./run.sh \
  --dataset acon96/Home-Assistant-Requests-V2 \
  --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit \
  --name Home
```

The `--path` argument is resolved in this order:
- Absolute path → used as-is
- `~/...` → expanded to home directory
- `./...` or `../...` → relative to current directory
- Otherwise → relative to `~/.lmstudio/models/`

The fused model name is automatically derived from the base model directory by inserting `--name` before the parameter count:
```
Qwen3-4b-Instruct-2507-MLX-8bit  →  Qwen3-Home-4b-Instruct-2507-MLX-8bit
```

### Memory-constrained hardware (e.g. 8B+ models)
```bash
./run.sh \
  --dataset acon96/Home-Assistant-Requests-V2 \
  --path lmstudio-community/Qwen3-8B-MLX-8bit \
  --name Home \
  --batch-size 1
```

### Dataset only (no training)
```bash
./run.sh \
  --dataset acon96/Home-Assistant-Requests-V2 \
  --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit \
  --name Home \
  --skip-train \
  --skip-fuse
```

### Fusion only (after training)
```bash
./run.sh \
  --dataset acon96/Home-Assistant-Requests-V2 \
  --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit \
  --name Home \
  --skip-train
```

The best checkpoint is auto-detected from `best_iter.txt`. Use `--best-iter N` to override.

### Stop and restart a pipeline

If a pipeline is already running, `run.sh` will refuse to start. Use `--force` to stop it automatically:
```bash
./run.sh --force \
  --dataset acon96/Home-Assistant-Requests-V2 \
  --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit \
  --name Home
```

### Reconnecting after SSH disconnect
```bash
./attach.sh
```

Shows the last 20 lines of the most recent log and follows it live if a pipeline is still running. Press `Ctrl+C` to detach without stopping the pipeline.

### Stopping a pipeline
```bash
./terminate.sh
```

Asks for confirmation before stopping. Use `--force` or `-f` to skip confirmation:
```bash
./terminate.sh --force
```

---

## All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset`, `-d` | *(required)* | HuggingFace dataset ID |
| `--path`, `-p` | *(required)* | Base model path |
| `--name` | `tuned` | Name inserted in fused model directory |
| `--fused-base` | `.` | Base directory for the fused model output |
| `--force`, `-f` | `false` | Stop any running pipeline before starting |
| `--skip-train` | `false` | Skip fine-tuning |
| `--skip-fuse` | `false` | Skip adapter fusion |
| `--best-iter` | auto | Checkpoint iter to use for fusion |
| `--iters` | `1000` | Training iterations |
| `--batch-size` | `2` | Batch size |
| `--num-layers` | `16` | Number of layers to fine-tune |
| `--learning-rate` | `1e-5` | Learning rate |
| `--max-seq-length` | `8192` | Maximum sequence length |
| `--lora-rank` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--patience` | `2` | Early stopping patience (in validations) |

---

## Cache Structure

All cache and working files are stored under `~/.mlx-fine-tune/`:
```
~/.mlx-fine-tune/
  venv/                              ← Python virtual environment
  logs/
    run_YYYYMMDD_HHMMSS_PID.log      ← one log file per run
    run_YYYYMMDD_HHMMSS_PID.pid      ← PID file for process management
  cache/
    parquet/
      <dataset_hash>/                ← downloaded .parquet files
    raw/
      <dataset_hash>/
        rows.jsonl                   ← flat JSONL (all rows, no template)
    template/
      <dataset_hash>_<tmpl_hash>/    ← train / valid / test .jsonl
    lora_config.yaml                 ← generated LoRA config
  work/
    data/
      <dataset_hash>_<tmpl_hash>/    ← copy of dataset used for training
    adapters/
      <model>_<dataset_hash>_<tmpl_hash>/
        0000100_adapters.safetensors
        0000200_adapters.safetensors
        ...
        adapters.safetensors         ← best checkpoint (copied before fusion)
        adapter_config.json
        best_iter.txt                ← best checkpoint iter (written by early stopping)
```

### Cache keys

| Cache | Key |
|-------|-----|
| Parquet files | SHA256(dataset\_id)[:12] |
| Raw JSONL | SHA256(dataset\_id)[:12] |
| Masked JSONL | SHA256(dataset\_id)[:12] + SHA256(chat\_template)[:12] |
| Adapters | model\_name + SHA256(dataset\_id)[:12] + SHA256(chat\_template)[:12] |

Two models sharing the same chat template (e.g. different Qwen3 sizes) will reuse the same masked JSONL cache.

---

## Loading the Fused Model in LM Studio

Move the fused model directory into LM Studio's model folder under a publisher name of your choice:
```bash
mkdir -p ~/.lmstudio/models/my-publisher
mv ./Qwen3-Home-4b-Instruct-2507-MLX-8bit ~/.lmstudio/models/my-publisher/
```

LM Studio will detect it automatically — no download required.