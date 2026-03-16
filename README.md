# MLX Fine-Tune

A pipeline for fine-tuning and evaluating MLX language models on HuggingFace
datasets, optimized for Apple Silicon (M1/M2/M3). Features automatic caching
at every stage, early stopping, LoRA adapter fusion, and structured evaluation
with accuracy metrics.

---

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- [LM Studio](https://lmstudio.ai) (optional, for model management)

No manual dependency installation required — the scripts handle everything
automatically via a dedicated virtual environment.

---

## Installation
```bash
chmod +x run.sh evaluate.sh attach.sh terminate.sh
```

---

## Repository structure
```
mlx-fine-tune/
  run.sh               ← launch a fine-tuning pipeline
  evaluate.sh          ← launch an evaluation pipeline
  attach.sh            ← reconnect to a running pipeline
  terminate.sh         ← stop a running pipeline
  pipeline.py          ← fine-tuning logic
  evaluate.py          ← evaluation logic
  lib/
    utils.py           ← shared utilities
    dataset.py         ← parquet download and cache management
    model.py           ← model path resolution and tokenizer loading
    template.py        ← Jinja2 chat template rendering
  formats/
    registry.py        ← format registry
    acon96_v2.py       ← acon96/Home-Assistant-Requests-V2
    allenporter_fc.py  ← allenporter/assist-llm-function-calling
    allenporter_msg.py ← allenporter/assist-llm-function-calling-messages
```

---

## Pipeline Overview
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
        ↓
   Evaluation Report
```

**Stage 1 — Parquet download**: fetches `.parquet` files from HuggingFace,
validates by file size.

**Stage 2 — Raw JSONL**: converts Parquet row-by-row via `pyarrow`.
Shared between training and evaluation — downloaded once per dataset.

**Stage 3 — Chat template application**: applies the model's Jinja2 chat
template to produce masked training examples. Only `train_on_turn: true`
turns contribute to the loss. Cache is keyed on dataset + chat template —
models with the same template share this cache.

**Stage 4 — LoRA fine-tuning**: runs `mlx_lm lora` with real-time early
stopping. Best checkpoint saved to `best_iter.txt`. Auto-resumes if a
previous run exists.

**Stage 5 — Adapter fusion**: merges the best LoRA adapter into the base
model via `mlx_lm fuse`.

**Stage 6 — Evaluation**: loads the fused model once via the `mlx_lm`
Python API, runs inference on each example, and scores tool call accuracy.

---

## Supported Dataset Formats

| Format | Dataset | Training | Evaluation |
|--------|---------|----------|------------|
| `acon96-v2` | acon96/Home-Assistant-Requests-V2 | ✅ | ✅ |
| `allenporter-fc` | allenporter/assist-llm-function-calling | ❌ | ✅ |
| `allenporter-msg` | allenporter/assist-llm-function-calling-messages | ✅ | ✅ |

The format is specified with `--dataset-format` on the first run and
auto-detected from cache on subsequent runs.

---

## Usage

### Fine-tuning
```bash
# First run — format required
./run.sh \
  --dataset acon96/Home-Assistant-Requests-V2 \
  --dataset-format acon96-v2 \
  --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit \
  --name Home

# Subsequent runs — format auto-detected
./run.sh \
  --dataset acon96/Home-Assistant-Requests-V2 \
  --path lmstudio-community/Qwen3-8B-MLX-8bit \
  --name Home \
  --batch-size 1
```

### Evaluation
```bash
./evaluate.sh \
  --model lalexdotcom/Qwen3-Home-4b-Instruct-2507-MLX-8bit \
  --dataset allenporter/assist-llm-function-calling \
  --dataset-format allenporter-fc
```

### Dataset only
```bash
./run.sh \
  --dataset acon96/Home-Assistant-Requests-V2 \
  --dataset-format acon96-v2 \
  --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit \
  --name Home \
  --skip-train --skip-fuse
```

### Fusion only
```bash
./run.sh \
  --dataset acon96/Home-Assistant-Requests-V2 \
  --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit \
  --name Home \
  --skip-train
```

### Stop and restart
```bash
./run.sh --force \
  --dataset acon96/Home-Assistant-Requests-V2 \
  --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit \
  --name Home
```

### Reconnect after SSH disconnect
```bash
./attach.sh
```

### Stop a pipeline
```bash
./terminate.sh
./terminate.sh --force
```

---

## All Options

### run.sh

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset`, `-d` | required | HuggingFace dataset ID |
| `--dataset-format` | auto | Format (auto-detected after first use) |
| `--path`, `-p` | required | Model path |
| `--name` | `tuned` | Inserted before param count in fused model name |
| `--fused-base` | `.` | Base directory for fused model output |
| `--force`, `-f` | false | Stop running pipeline before starting |
| `--skip-train` | false | Skip fine-tuning |
| `--skip-fuse` | false | Skip adapter fusion |
| `--best-iter` | auto | Checkpoint iter to use for fusion |
| `--iters` | 1000 | Training iterations |
| `--batch-size` | 2 | Batch size (use 1 for 8B+ models) |
| `--num-layers` | 16 | Layers to fine-tune |
| `--learning-rate` | 1e-5 | Learning rate |
| `--max-seq-length` | 8192 | Max sequence length |
| `--lora-rank` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha |
| `--patience` | 2 | Early stopping patience (in validations) |

### evaluate.sh

| Option | Default | Description |
|--------|---------|-------------|
| `--model`, `-m` | required | Model path (same resolution as --path) |
| `--dataset`, `-d` | required | HuggingFace dataset ID |
| `--dataset-format` | auto | Format (auto-detected after first use) |
| `--max-tokens` | 512 | Max tokens to generate per example |
| `--limit` | all | Limit to first N examples |

---

## Cache Structure
```
~/.mlx-fine-tune/
  venv/                                     ← Python virtual environment
  logs/
    run_YYYYMMDD_HHMMSS_PID.log             ← fine-tuning run log
    eval_YYYYMMDD_HHMMSS_PID.log            ← evaluation run log
    *.pid                                   ← PID files
  cache/
    dataset_formats.json                    ← dataset hash → format name
    parquet/<dataset_hash>/                 ← downloaded .parquet files
    raw/<dataset_hash>/rows.jsonl           ← flat JSONL (all rows)
    template/<dataset_hash>_<tmpl_hash>/    ← train / valid / test .jsonl
    lora_config.yaml                        ← generated LoRA config
  work/
    data/<dataset_hash>_<tmpl_hash>/        ← training input copy
    adapters/<model>_<dataset_hash>_<tmpl_hash>/
      0000100_adapters.safetensors
      ...
      adapters.safetensors                  ← best checkpoint for fusion
      adapter_config.json
      best_iter.txt                         ← best iter (written by early stopping)
  evals/<model>_<dataset_hash>_<run_id>/
    results.json                            ← all examples with scores
    summary.json                            ← aggregated metrics
    report.md                               ← human-readable report
```

### Cache keys

| Cache | Key |
|-------|-----|
| Parquet | SHA256(dataset\_id)[:12] |
| Raw JSONL | SHA256(dataset\_id)[:12] |
| Masked JSONL | SHA256(dataset\_id)[:12] + SHA256(chat\_template)[:12] |
| Adapters | model\_name + SHA256(dataset\_id)[:12] + SHA256(chat\_template)[:12] |

---

## Loading a Fused Model in LM Studio
```bash
mkdir -p ~/.lmstudio/models/lalexdotcom
mv ./Qwen3-Home-4b-Instruct-2507-MLX-8bit ~/.lmstudio/models/lalexdotcom/
```

LM Studio detects the model automatically — no download required.