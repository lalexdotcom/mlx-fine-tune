# MLX Fine-Tune — Copilot Instructions

## Project overview

CLI pipeline for fine-tuning MLX language models on HuggingFace datasets, optimized for Apple Silicon (M1/M2/M3). The pipeline covers the full workflow: dataset download, conversion, LoRA fine-tuning with early stopping, and adapter fusion.

## Repository structure
```
mlx-fine-tune/
  run.sh            ← main entry point (launch a pipeline)
  attach.sh         ← reconnect to a running pipeline
  terminate.sh      ← stop a running pipeline (--force / -f to skip confirmation)
  pipeline.py       ← full pipeline logic (Python)
  README.md
  .github/
    copilot-instructions.md
```

## Tech stack

- **Shell**: bash scripts for process management (nohup, PID files, trap)
- **Python**: pipeline logic, all dependencies managed in a venv at `~/.mlx-fine-tune/venv/`
- **Key Python libs**: `pyarrow` (Parquet), `jinja2` (chat template rendering), `mlx_lm` (fine-tuning + fusion), `pyyaml` (LoRA config)
- **TypeScript**: not used — the project was started in TS then migrated to Python

## Conventions

- All user-facing messages in **English**
- Python type hints used throughout (`list[Path]`, `dict`, etc.)
- No hardcoded paths — everything derived from `--work-dir` passed by `run.sh`
- Cache directory: `~/.mlx-fine-tune/` (defined once in `run.sh`, passed to `pipeline.py` via `--work-dir`)
- Shell scripts use `set -e` — all commands must succeed or the script exits
- `--force` / `-f` is the convention for skipping confirmations across all scripts
- New CLI parameters must be added to both `parse_args()` and `apply_overrides()` in `pipeline.py`
- Any new cache level must follow the existing key convention: `SHA256(value)[:12]`

## Code structure (pipeline.py)

The file is organized in this order — maintain this structure when adding features:

1. **Global path variables** (`VENV_PYTHON`, `CACHE_DIR`, `WORK_DIR`) — all `None` at import, initialized by `init_dirs()` after `parse_args()`
2. **Constants** (`LM_STUDIO_MODELS_DIR`, `HF_PARQUET_API`)
3. **Config dicts** (`LORA_CONFIG`, `LORA_YAML_CONFIG`) — all tunable parameters live here
4. **Utils** — pure functions with no side effects (`format_bytes`, `count_jsonl_lines`, `short_hash`, etc.)
5. **Cache key helpers** — functions that return `Path` objects (`parquet_cache_dir`, `raw_jsonl_path`, etc.)
6. **Args** — `parse_args()` then `apply_overrides()`
7. **Pipeline stages** — one function per stage, in order
8. **`run_command()`** — thin wrapper around `subprocess.run`
9. **`run_fine_tuning()`** — uses `subprocess.Popen` for live output + early stopping
10. **`run_fuse()`**
11. **`main()`**

## Cache architecture

Four levels of cache, all under `~/.mlx-fine-tune/`:

| Level | Path | Key |
|-------|------|-----|
| Parquet files | `cache/parquet/<dataset_hash>/` | SHA256(dataset_id)[:12] |
| Raw JSONL | `cache/raw/<dataset_hash>/rows.jsonl` | SHA256(dataset_id)[:12] |
| Masked JSONL | `cache/template/<dataset_hash>_<tmpl_hash>/` | dataset_hash + SHA256(chat_template)[:12] |
| LoRA adapters | `work/adapters/<model>_<dataset_hash>_<tmpl_hash>/` | model_name + dataset_hash + tmpl_hash |

Two models with the same chat template reuse the same masked JSONL cache.

When adding a new cache level:
- Add a helper function in the "Cache key helpers" section
- Add a validity check function (same pattern as `raw_cache_valid`, `template_cache_valid`)
- Document the key in this file

## Pipeline stages (pipeline.py)

1. **Download** — fetches `.parquet` files from HF Datasets API, checks size against remote for cache validity
2. **Parquet → raw JSONL** — row-group streaming via `pyarrow`; cache valid if line count matches parquet row count
3. **Chat template application** — Jinja2 rendering of `tokenizer_config.json` chat_template; only `train_on_turn: true` turns are included in the loss; diff-based approach (render N messages, subtract render of N-1)
4. **LoRA fine-tuning** — `mlx_lm lora` with early stopping via `subprocess.Popen`; best iter saved to `best_iter.txt`; auto-resumes from last checkpoint
5. **Adapter fusion** — `mlx_lm fuse`; copies best checkpoint to `adapters.safetensors` before fusing

## Key design decisions

- `--work-dir` is the single source of truth for all paths — never hardcoded in `pipeline.py`
- Early stopping parses `Iter X: Val loss Y` lines from mlx_lm stdout in real time
- `best_iter.txt` persists the best checkpoint so `--skip-train` + fusion works without `--best-iter`
- `nohup` + PID files allow SSH disconnect without killing the pipeline
- `run.sh` prevents concurrent runs; `--force` stops any existing pipeline before starting
- `subprocess.Popen` (not `run`) is used for fine-tuning to enable live output parsing
- Chat template is rendered via Jinja2 directly from `tokenizer_config.json` — no hardcoded format detection

## Adding a new CLI parameter

1. Add to `parse_args()` with a descriptive `help=` string including the default value
2. Add to `apply_overrides()` with a `if args.xxx is not None` guard
3. If it's a LoRA parameter, add to `LORA_CONFIG` or `LORA_YAML_CONFIG`
4. Update the parameters table in `README.md` and in this file

## Adding a new pipeline stage

1. Add a stage function following the existing pattern (validate cache → process → write)
2. Add corresponding cache key helper and validity check
3. Call it in `main()` in the correct order
4. Add `--skip-xxx` parameter if the stage should be skippable
5. Update both `README.md` and this file

## CLI parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset`, `-d` | required | HuggingFace dataset ID |
| `--path`, `-p` | required | Model path (relative to `~/.lmstudio/models/`, absolute, `~/`, `./`, `../`) |
| `--name` | `tuned` | Inserted before param count in fused model name |
| `--fused-base` | `.` | Output directory for fused model |
| `--force`, `-f` | false | Stop running pipeline before starting |
| `--skip-train` | false | Skip fine-tuning |
| `--skip-fuse` | false | Skip adapter fusion |
| `--best-iter` | auto | Override checkpoint iter for fusion |
| `--iters` | 1000 | Training iterations |
| `--batch-size` | 2 | Batch size (use 1 for 8B+ models) |
| `--num-layers` | 16 | LoRA layers |
| `--learning-rate` | 1e-5 | Learning rate |
| `--max-seq-length` | 8192 | Max sequence length |
| `--lora-rank` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha |
| `--patience` | 2 | Early stopping patience (in validations) |

## Typical commands
```bash
# Full pipeline — 4B model
./run.sh --dataset acon96/Home-Assistant-Requests-V2 --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit --name Home

# Full pipeline — 8B+ model (reduced batch size)
./run.sh --dataset acon96/Home-Assistant-Requests-V2 --path lmstudio-community/Qwen3-8B-MLX-8bit --name Home --batch-size 1

# Fusion only (after training)
./run.sh --dataset acon96/Home-Assistant-Requests-V2 --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit --name Home --skip-train

# Stop and restart
./run.sh --force --dataset acon96/Home-Assistant-Requests-V2 --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit --name Home

# Reconnect after SSH disconnect
./attach.sh

# Stop running pipeline
./terminate.sh --force
```

## LM Studio integration

Models are stored under `~/.lmstudio/models/<publisher>/<model-name>/`.
After fusion, move the output directory:
```bash
mkdir -p ~/.lmstudio/models/lalexdotcom
mv ./Qwen3-Home-4b-Instruct-2507-MLX-8bit ~/.lmstudio/models/lalexdotcom/
```