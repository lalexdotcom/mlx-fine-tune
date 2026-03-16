# MLX Fine-Tune — Copilot Instructions

## Project overview

CLI pipeline for fine-tuning and evaluating MLX language models on HuggingFace
datasets, optimized for Apple Silicon (M1/M2/M3). The pipeline covers the full
workflow: dataset download, conversion, LoRA fine-tuning with early stopping,
adapter fusion, and model evaluation.

## Repository structure
```
mlx-fine-tune/
  run.sh            ← launch a fine-tuning pipeline
  evaluate.sh       ← launch an evaluation pipeline
  attach.sh         ← reconnect to a running pipeline
  terminate.sh      ← stop a running pipeline (--force / -f to skip confirmation)
  pipeline.py       ← fine-tuning pipeline logic
  evaluate.py       ← evaluation pipeline logic
  lib/
    __init__.py
    utils.py        ← shared utilities (format_bytes, count_jsonl_lines, short_hash)
    dataset.py      ← parquet download, cache management, format registry
    model.py        ← resolve_model_path, read_tokenizer_config, get_token_string
    template.py     ← Jinja2 chat template rendering, diff-based turn isolation
  formats/
    __init__.py     ← MaskedExample, EvalExample, FormatError dataclasses
    registry.py     ← format name → module mapping, load_format(), list_formats()
    acon96_v2.py    ← acon96/Home-Assistant-Requests-V2 (messages + train_on_turn)
    allenporter_fc.py  ← allenporter/assist-llm-function-calling (eval only)
    allenporter_msg.py ← allenporter/assist-llm-function-calling-messages
  README.md
  .github/
    copilot-instructions.md
```

## Tech stack

- **Shell**: bash scripts for process management (nohup, PID files, trap)
- **Python**: all pipeline logic, managed in a venv at `~/.mlx-fine-tune/venv/`
- **Key Python libs**: `pyarrow` (Parquet), `jinja2` (chat template rendering),
  `mlx_lm` (fine-tuning, fusion, inference), `pyyaml` (LoRA config)

## Conventions

- All user-facing messages and comments in **English**
- Python type hints used throughout (`list[Path]`, `dict`, etc.)
- No hardcoded paths — everything derived from `--work-dir` passed by shell scripts
- Cache directory: `~/.mlx-fine-tune/` (defined once in shell scripts)
- Shell scripts use `set -e` — all commands must succeed or the script exits
- `--force` / `-f` is the convention for skipping confirmations across all scripts
- New CLI parameters must be added to both `parse_args()` and `apply_overrides()`
- Any new cache level must follow the existing key convention: `SHA256(value)[:12]`
- All functions must have docstrings explaining purpose, args, and return value

## Code structure (pipeline.py and evaluate.py)

Both files follow this organization — maintain it when adding features:

1. **Global path variables** — all `None` at import, initialized by `init_dirs()`
2. **Constants** — `LM_STUDIO_MODELS_DIR`, API URLs
3. **Config dicts** — all tunable parameters (`LORA_CONFIG`, `GENERATE_CONFIG`)
4. **Cache key helpers** — functions returning `Path` objects
5. **Utils** — pure helper functions
6. **Args** — `parse_args()` then `apply_overrides()`
7. **Pipeline stages** — one function per stage, in order
8. **`run_command()`** — thin wrapper around `subprocess.run`
9. **Main loop function** — `run_fine_tuning()` or `run_evaluation()`
10. **`main()`**

## Cache architecture

All cache under `~/.mlx-fine-tune/`:

| Level | Path | Key |
|-------|------|-----|
| Parquet files | `cache/parquet/<dataset_hash>/` | SHA256(dataset_id)[:12] |
| Raw JSONL | `cache/raw/<dataset_hash>/rows.jsonl` | SHA256(dataset_id)[:12] |
| Masked JSONL | `cache/template/<dataset_hash>_<tmpl_hash>/` | dataset_hash + SHA256(chat_template)[:12] |
| LoRA adapters | `work/adapters/<model>_<dataset_hash>_<tmpl_hash>/` | model_name + dataset_hash + tmpl_hash |
| Eval results | `evals/<model>_<dataset_hash>_<run_id>/` | model_name + dataset_hash + run_id |
| Format registry | `cache/dataset_formats.json` | dataset_hash → format name |

Two models with the same chat template reuse the same masked JSONL cache.
The raw JSONL cache is shared between training and evaluation pipelines.

When adding a new cache level:
- Add a helper function in `lib/dataset.py`
- Add a validity check function
- Document the key in this file and in README.md

## Format system

Each dataset format is a Python module in `formats/` exposing two functions:
```python
def convert_for_training(example: dict, tokenizer_config: dict) -> list[MaskedExample]: ...
def convert_for_eval(example: dict, tokenizer_config: dict) -> EvalExample | None: ...
```

Formats are registered in `formats/registry.py` and resolved by name via
`--dataset-format`. The format is auto-detected from `cache/dataset_formats.json`
on subsequent runs — no need to pass `--dataset-format` again for known datasets.

### Supported formats

| Format | Dataset | Training | Evaluation |
|--------|---------|----------|------------|
| `acon96-v2` | acon96/Home-Assistant-Requests-V2 | ✅ | ✅ |
| `allenporter-fc` | allenporter/assist-llm-function-calling | ❌ | ✅ |
| `allenporter-msg` | allenporter/assist-llm-function-calling-messages | ✅ | ✅ |

### Adding a new format

1. Create `formats/<name>.py` with `convert_for_training()` and `convert_for_eval()`
2. Add entry to `REGISTRY` in `formats/registry.py`
3. Add row to the supported formats table above and in README.md

## Pipeline stages (pipeline.py)

1. **Format resolution** — load format module, fail early with helpful message
2. **Download** — fetch `.parquet` files from HF Datasets API, validate by size
3. **Parquet → raw JSONL** — row-group streaming via `pyarrow`
4. **Chat template application** — Jinja2 diff-based rendering, only `train_on_turn=True` turns included
5. **LoRA fine-tuning** — `mlx_lm lora` with real-time early stopping via `subprocess.Popen`; saves `best_iter.txt`; auto-resumes from last checkpoint
6. **Adapter fusion** — `mlx_lm fuse`; copies best checkpoint before fusing

## Evaluation stages (evaluate.py)

1. **Format resolution** — same as pipeline.py
2. **Download + raw JSONL** — reuses shared cache from training pipeline
3. **Model loading** — `mlx_lm.load()` once; model stays in memory for all examples
4. **Inference loop** — `mlx_lm.generate()` per example, temperature=0 for determinism
5. **Scoring** — tool name match, argument match, false positive/negative detection
6. **Report** — `results.json`, `summary.json`, `report.md` in `evals/`

## Model path resolution

Used in both `pipeline.py` and `evaluate.py` via `lib/model.resolve_model_path()`.
Always reuse this function — never reimplement it.

Resolution order:
- `~/...` → home directory
- absolute → as-is
- `./` or `../` → relative to cwd
- otherwise → relative to `~/.lmstudio/models/`

Fine-tuned models in LM Studio: `lalexdotcom/<model-name>`

## Key design decisions

- `--work-dir` is the single source of truth for all paths
- Early stopping parses `Iter X: Val loss Y` from mlx_lm stdout in real time
- `best_iter.txt` persists best checkpoint for use with `--skip-train`
- `nohup` + PID files allow SSH disconnect without killing the pipeline
- `subprocess.Popen` for fine-tuning (live output); Python API for evaluation (speed)
- Chat template rendered via Jinja2 from `tokenizer_config.json` — no hardcoded formats
- Format auto-detection via `cache/dataset_formats.json` — no need to repeat `--dataset-format`

## CLI parameters

### run.sh / pipeline.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset`, `-d` | required | HuggingFace dataset ID |
| `--dataset-format` | auto | Format converter (auto-detected after first use) |
| `--path`, `-p` | required | Model path |
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

### evaluate.sh / evaluate.py

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model`, `-m` | required | Model path (same resolution as --path) |
| `--dataset`, `-d` | required | HuggingFace dataset ID |
| `--dataset-format` | auto | Format converter (auto-detected after first use) |
| `--max-tokens` | 512 | Max tokens to generate per example |
| `--limit` | all | Limit evaluation to first N examples |

## Typical commands
```bash
# Fine-tuning — 4B model (first run, format required)
./run.sh --dataset acon96/Home-Assistant-Requests-V2 \
         --dataset-format acon96-v2 \
         --path lmstudio-community/Qwen3-4b-Instruct-2507-MLX-8bit \
         --name Home

# Fine-tuning — subsequent runs (format auto-detected)
./run.sh --dataset acon96/Home-Assistant-Requests-V2 \
         --path lmstudio-community/Qwen3-8B-MLX-8bit \
         --name Home --batch-size 1

# Evaluation
./evaluate.sh --model lalexdotcom/Qwen3-Home-4b-Instruct-2507-MLX-8bit \
              --dataset allenporter/assist-llm-function-calling \
              --dataset-format allenporter-fc

# Reconnect after SSH disconnect
./attach.sh

# Stop running pipeline
./terminate.sh --force
```

## LM Studio integration

Models stored under `~/.lmstudio/models/<publisher>/<model-name>/`.
Publisher used for fine-tuned models: `lalexdotcom`

Current fine-tuned models:
- `lalexdotcom/Qwen3-Home-4b-Instruct-2507-MLX-8bit` (best iter: 400)
- `lalexdotcom/Qwen3-Home-8B-MLX-8bit` (best iter: 200)
- `lalexdotcom/Qwen3-Home-14B-MLX-8bit` (in progress)