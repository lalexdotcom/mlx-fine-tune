# pipeline.py
#
# Main fine-tuning pipeline for MLX language models.
# Orchestrates dataset download, chat template conversion, LoRA fine-tuning,
# and adapter fusion. All paths are derived from --work-dir to avoid
# hardcoded paths.

import argparse
import json
import re
import shutil
import subprocess
import yaml
from pathlib import Path

from lib.utils import short_hash, count_jsonl_lines
from lib.dataset import (
    download_parquet_files,
    parquet_to_raw_jsonl,
    raw_jsonl_path,
    template_cache_dir,
    template_cache_valid,
    resolve_dataset_format,
)
from lib.model import resolve_model_path, read_tokenizer_config
from formats.registry import load_format

# ─── Paths derived from --work-dir ───────────────────────────────────────────
# All None at import time, initialized by init_dirs() after parse_args()

VENV_PYTHON = None
CACHE_DIR   = None
WORK_DIR    = None


def init_dirs(work_dir: str) -> None:
    """Initialize global path variables from the work directory.
    Called once at startup after parsing CLI arguments.
    """
    global VENV_PYTHON, CACHE_DIR, WORK_DIR
    base        = Path(work_dir)
    VENV_PYTHON = base / "venv" / "bin" / "python3"
    CACHE_DIR   = base / "cache"
    WORK_DIR    = base / "work"

# ─── Fine-tuning config ───────────────────────────────────────────────────────
# All tunable parameters live here — override via CLI with apply_overrides()

LORA_CONFIG = {
    "iters": 1000,
    "batch_size": 2,
    "num_layers": 16,
    "learning_rate": 1e-5,
    "grad_checkpoint": True,
    "max_seq_length": 8192,
    "patience": 2,
}

LORA_YAML_CONFIG = {
    "lora_parameters": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "scale": 10.0,
    }
}

# ─── Cache key helpers ────────────────────────────────────────────────────────

def adapters_dir(model_name: str, dataset_hash: str, template_hash: str) -> Path:
    """Return the directory for LoRA adapter checkpoints.
    Keyed on model name + dataset hash + template hash to avoid mixing
    adapters from different training runs.
    """
    return WORK_DIR / "adapters" / f"{model_name}_{dataset_hash}_{template_hash}"


def data_dir(dataset_hash: str, template_hash: str) -> Path:
    """Return the working data directory for a specific dataset + template combo.
    This is a copy of the template cache used as mlx_lm training input.
    """
    return WORK_DIR / "data" / f"{dataset_hash}_{template_hash}"

# ─── Utils ────────────────────────────────────────────────────────────────────

def build_fused_path(model_path: Path, fused_base: str, name: str) -> Path:
    """Derive the fused model output path by inserting --name before the
    parameter count in the model directory name.
    Example: Qwen3-4b-Instruct + name=Home → Qwen3-Home-4b-Instruct
    Falls back to appending the name if no parameter count is found.
    """
    dir_name = model_path.name
    match = re.search(r"(\d+\.?\d*[bBmM])", dir_name)
    if match:
        insert_pos = match.start()
        before = dir_name[:insert_pos].rstrip("-_")
        after = dir_name[insert_pos:]
        fused_name = f"{before}-{name}-{after}"
    else:
        fused_name = f"{dir_name}-{name}"
    return Path(fused_base) / fused_name

# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MLX LLM fine-tuning pipeline")
    p.add_argument("--work-dir", required=True,
                   help="Base working directory (set by run.sh)")
    p.add_argument("--dataset", "-d", required=True,
                   help="HuggingFace dataset ID (e.g. acon96/Home-Assistant-Requests-V2)")
    p.add_argument("--dataset-format", default=None,
                   help="Dataset format converter (e.g. acon96-v2). "
                        "Auto-detected from cache if previously used with this dataset.")
    p.add_argument("--path", "-p", required=True,
                   help="Model path (relative to ~/.lmstudio/models/, absolute, ~/... or ./local)")
    p.add_argument("--name", type=str, default="tuned",
                   help="Name to insert in the fused model directory (default: tuned)")
    p.add_argument("--fused-base", default=".",
                   help="Base directory for the fused model (default: current directory)")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip fine-tuning")
    p.add_argument("--skip-fuse", action="store_true",
                   help="Skip adapter fusion")
    p.add_argument("--best-iter", type=int, default=None,
                   help="Checkpoint iter to use for fusion (default: auto-detected)")
    p.add_argument("--run-id", type=str, default="manual",
                   help="Run ID for logging (injected by run.sh)")
    p.add_argument("--iters", type=int, default=None,
                   help=f"Training iterations (default: {LORA_CONFIG['iters']})")
    p.add_argument("--batch-size", type=int, default=None,
                   help=f"Batch size (default: {LORA_CONFIG['batch_size']})")
    p.add_argument("--num-layers", type=int, default=None,
                   help=f"Layers to fine-tune (default: {LORA_CONFIG['num_layers']})")
    p.add_argument("--learning-rate", type=float, default=None,
                   help=f"Learning rate (default: {LORA_CONFIG['learning_rate']})")
    p.add_argument("--max-seq-length", type=int, default=None,
                   help=f"Max sequence length (default: {LORA_CONFIG['max_seq_length']})")
    p.add_argument("--lora-rank", type=int, default=None,
                   help=f"LoRA rank (default: {LORA_YAML_CONFIG['lora_parameters']['rank']})")
    p.add_argument("--lora-alpha", type=int, default=None,
                   help=f"LoRA alpha (default: {LORA_YAML_CONFIG['lora_parameters']['alpha']})")
    p.add_argument("--patience", type=int, default=None,
                   help=f"Early stopping patience (default: {LORA_CONFIG['patience']})")
    return p.parse_args()


def apply_overrides(args) -> None:
    """Apply CLI overrides to the global LORA_CONFIG and LORA_YAML_CONFIG dicts.
    Only overrides values that were explicitly passed on the command line.
    """
    if args.iters is not None:
        LORA_CONFIG["iters"] = args.iters
    if args.batch_size is not None:
        LORA_CONFIG["batch_size"] = args.batch_size
    if args.num_layers is not None:
        LORA_CONFIG["num_layers"] = args.num_layers
    if args.learning_rate is not None:
        LORA_CONFIG["learning_rate"] = args.learning_rate
    if args.max_seq_length is not None:
        LORA_CONFIG["max_seq_length"] = args.max_seq_length
    if args.lora_rank is not None:
        LORA_YAML_CONFIG["lora_parameters"]["rank"] = args.lora_rank
    if args.lora_alpha is not None:
        LORA_YAML_CONFIG["lora_parameters"]["alpha"] = args.lora_alpha
    if args.patience is not None:
        LORA_CONFIG["patience"] = args.patience

# ─── Dataset conversion ───────────────────────────────────────────────────────

def copy_template_cache(
    dataset_hash: str,
    tmpl_hash: str,
    out_dir: Path,
) -> None:
    """Copy the cached masked JSONL splits to the working data directory
    used by mlx_lm as training input.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = template_cache_dir(CACHE_DIR, dataset_hash, tmpl_hash)
    for split in ["train", "valid", "test"]:
        src = src_dir / f"{split}.jsonl"
        dst = out_dir / f"{split}.jsonl"
        shutil.copyfile(src, dst)
        lines = count_jsonl_lines(dst)
        print(f"  {dst} — {lines} examples (from cache)")


def raw_jsonl_to_masked(
    dataset_hash: str,
    tokenizer_config: dict,
    tmpl_hash: str,
    fmt_module,
) -> None:
    """Convert raw JSONL to masked training examples using the given format module.
    Streams line by line to keep memory usage low.
    Results are written to the template cache directory.
    """
    raw_path = raw_jsonl_path(CACHE_DIR, dataset_hash)
    out_dir = template_cache_dir(CACHE_DIR, dataset_hash, tmpl_hash)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = count_jsonl_lines(raw_path)
    print(f"  {total} raw examples to convert...")

    writers = {
        "train": open(out_dir / "train.jsonl", "w"),
        "valid": open(out_dir / "valid.jsonl", "w"),
        "test":  open(out_dir / "test.jsonl",  "w"),
    }
    counts = {"train": 0, "valid": 0, "test": 0}
    index = 0

    def assign_split(idx: int, total: int) -> str:
        """Assign an example to train/valid/test split by position ratio."""
        r = idx / total
        if r < 0.8: return "train"
        if r < 0.9: return "valid"
        return "test"

    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except Exception:
                index += 1
                continue

            masked = fmt_module.convert_for_training(example, tokenizer_config)
            split = assign_split(index, total)
            index += 1

            for item in masked:
                writers[split].write(json.dumps({"text": item.text}) + "\n")
                counts[split] += 1

            if index % 1000 == 0:
                pct = int(index / total * 100)
                print(
                    f"\r  [{pct:3d}%] {index}/{total} — "
                    f"train: {counts['train']} "
                    f"valid: {counts['valid']} "
                    f"test: {counts['test']}   ",
                    end="", flush=True,
                )

    print(flush=True)
    for name, f in writers.items():
        f.close()
        print(f"  {out_dir}/{name}.jsonl — {counts[name]} examples")

# ─── Run command ──────────────────────────────────────────────────────────────

def run_command(cmd: list[str]) -> None:
    """Run a subprocess command and raise RuntimeError if it fails."""
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

# ─── Fine-tuning with early stopping ─────────────────────────────────────────

def run_fine_tuning(
    model_path: Path,
    out_data_dir: Path,
    adp_dir: Path,
) -> int:
    """Run LoRA fine-tuning via mlx_lm with real-time early stopping.
    Monitors validation loss from stdout and stops training when it has not
    improved for --patience consecutive validations.
    Saves best_iter.txt so --skip-train + fusion can find the best checkpoint.
    Returns the best iter number.
    """
    print("\n══════════════════════════════════════════")
    print("  Phase 1: LoRA Fine-tuning")
    print(f"  iters={LORA_CONFIG['iters']} batch={LORA_CONFIG['batch_size']} "
          f"layers={LORA_CONFIG['num_layers']} lr={LORA_CONFIG['learning_rate']} "
          f"max_seq={LORA_CONFIG['max_seq_length']} patience={LORA_CONFIG['patience']}")
    print(f"  rank={LORA_YAML_CONFIG['lora_parameters']['rank']} "
          f"alpha={LORA_YAML_CONFIG['lora_parameters']['alpha']}")
    print("══════════════════════════════════════════")

    adp_dir.mkdir(parents=True, exist_ok=True)

    lora_config_path = CACHE_DIR / "lora_config.yaml"
    with open(lora_config_path, "w") as f:
        yaml.dump(LORA_YAML_CONFIG, f)

    # Resume from last checkpoint if one exists
    checkpoints = sorted(adp_dir.glob("*_adapters.safetensors"))
    resume_args = []
    if checkpoints:
        last_checkpoint = checkpoints[-1]
        last_iter = int(last_checkpoint.stem.split("_")[0])
        print(f"  Resuming from checkpoint: iter {last_iter} ({last_checkpoint.name})")
        resume_args = ["--resume-adapter-file", str(last_checkpoint)]
    else:
        print(f"  Starting fresh fine-tuning")

    cmd = [
        str(VENV_PYTHON), "-m", "mlx_lm", "lora",
        "--model", str(model_path),
        "--train",
        "--data", str(out_data_dir),
        "--adapter-path", str(adp_dir),
        "--iters", str(LORA_CONFIG["iters"]),
        "--batch-size", str(LORA_CONFIG["batch_size"]),
        "--num-layers", str(LORA_CONFIG["num_layers"]),
        "--learning-rate", str(LORA_CONFIG["learning_rate"]),
        "--max-seq-length", str(LORA_CONFIG["max_seq_length"]),
        "-c", str(lora_config_path),
        *resume_args,
        *( ["--grad-checkpoint"] if LORA_CONFIG["grad_checkpoint"] else [] ),
    ]

    print(f"\n$ {' '.join(cmd)}\n")

    val_loss_re = re.compile(r"Iter (\d+): Val loss ([0-9.]+)")
    best_val_loss = float("inf")
    best_iter = 0
    no_improve_count = 0
    patience = LORA_CONFIG["patience"]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            m = val_loss_re.search(line)
            if m:
                current_iter = int(m.group(1))
                current_val_loss = float(m.group(2))
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_iter = current_iter
                    no_improve_count = 0
                    print(f"  ✓ New best val loss: {best_val_loss:.4f} at iter {best_iter}")
                else:
                    no_improve_count += 1
                    print(f"  ⚠ No improvement ({no_improve_count}/{patience}) — best: {best_val_loss:.4f} at iter {best_iter}")
                    if no_improve_count >= patience:
                        print(f"\n  🛑 Early stopping — best iter: {best_iter}, best val loss: {best_val_loss:.4f}")
                        proc.terminate()
                        proc.wait()
                        break
    except KeyboardInterrupt:
        print("\n  Interrupted — terminating process...")
        proc.terminate()
        proc.wait()

    if proc.returncode not in (0, -15, None):
        raise RuntimeError(f"Fine-tuning failed with exit code {proc.returncode}")

    # Persist best iter so --skip-train + fusion can find the right checkpoint
    best_iter_path = adp_dir / "best_iter.txt"
    best_iter_path.write_text(str(best_iter))
    print(f"  ✓ Best iter saved to {best_iter_path}")

    return best_iter

# ─── Adapter fusion ───────────────────────────────────────────────────────────

def run_fuse(
    model_path: Path,
    adp_dir: Path,
    fused_path: Path,
    best_iter: int,
) -> None:
    """Merge the best LoRA adapter into the base model using mlx_lm fuse.
    Copies the best checkpoint to adapters.safetensors before fusing.
    """
    print("\n══════════════════════════════════════════")
    print("  Phase 2: Adapter fusion")
    print("══════════════════════════════════════════")

    best_adapter = adp_dir / f"{best_iter:07d}_adapters.safetensors"
    target_adapter = adp_dir / "adapters.safetensors"

    if best_iter > 0 and best_adapter.exists():
        print(f"  Using checkpoint: iter {best_iter} ({best_adapter.name})")
        shutil.copyfile(best_adapter, target_adapter)
    elif target_adapter.exists():
        print(f"  Using existing adapters.safetensors")
    else:
        raise FileNotFoundError(f"No adapter file found in {adp_dir}")

    fused_path.mkdir(parents=True, exist_ok=True)

    run_command([
        str(VENV_PYTHON), "-m", "mlx_lm", "fuse",
        "--model", str(model_path),
        "--adapter-path", str(adp_dir),
        "--save-path", str(fused_path),
    ])

    print(f"\n✓ Fused model saved to: {fused_path}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    init_dirs(args.work_dir)
    apply_overrides(args)

    print(f"\n── Run {args.run_id} {'─' * 30}")

    # 1. Resolve model path and compute cache keys
    model_path = resolve_model_path(args.path)
    print(f"\n── Model ─────────────────────────────────")
    print(f"   {model_path}")

    tokenizer_config = read_tokenizer_config(model_path)
    d_hash = short_hash(args.dataset)
    t_hash = short_hash(tokenizer_config["chat_template"])

    # 2. Resolve dataset format — from CLI or cache
    try:
        dataset_format = resolve_dataset_format(
            CACHE_DIR, d_hash, args.dataset_format
        )
        fmt_module = load_format(dataset_format)
    except ValueError as e:
        print(f"\n❌ {e}")
        exit(1)

    print(f"   Dataset hash       : {d_hash}  ({args.dataset})")
    print(f"   Dataset format     : {dataset_format}")
    print(f"   Template hash      : {t_hash}")

    fused_path = build_fused_path(model_path, args.fused_base, args.name)
    print(f"   Fused model target : {fused_path}")

    # 3. Download parquet files
    print(f"\n── Dataset ───────────────────────────────")
    local_paths = download_parquet_files(args.dataset, d_hash, CACHE_DIR)

    # 4. Parquet → raw JSONL (cached per dataset, format-independent)
    parquet_to_raw_jsonl(CACHE_DIR, d_hash, local_paths)

    # 5. Raw JSONL → masked JSONL (cached per dataset + template)
    out_data_dir = data_dir(d_hash, t_hash)

    if template_cache_valid(CACHE_DIR, d_hash, t_hash):
        print(f"✓ Template cache hit ({d_hash}_{t_hash}) — skipping conversion")
        copy_template_cache(d_hash, t_hash, out_data_dir)
    else:
        print(f"\nApplying chat template (Jinja2) with format {dataset_format}...")
        raw_jsonl_to_masked(d_hash, tokenizer_config, t_hash, fmt_module)
        copy_template_cache(d_hash, t_hash, out_data_dir)

    # 6. Fine-tuning
    adp_dir = adapters_dir(model_path.name, d_hash, t_hash)
    best_iter = 0

    if not args.skip_train:
        best_iter = run_fine_tuning(model_path, out_data_dir, adp_dir)
    else:
        print("\n⏭  Fine-tuning skipped (--skip-train)")
        if args.best_iter is not None:
            best_iter = args.best_iter
            print(f"  Using specified checkpoint: iter {best_iter}")
        else:
            best_iter_path = adp_dir / "best_iter.txt"
            if best_iter_path.exists():
                best_iter = int(best_iter_path.read_text().strip())
                print(f"  Auto-detected best checkpoint: iter {best_iter} (from best_iter.txt)")
            else:
                checkpoints = sorted(adp_dir.glob("*_adapters.safetensors"))
                if checkpoints:
                    best_iter = int(checkpoints[-1].stem.split("_")[0])
                    print(f"  Auto-detected latest checkpoint: iter {best_iter}")
                else:
                    print(f"  No checkpoint found — will use adapters.safetensors directly")

    # 7. Fusion
    if not args.skip_fuse:
        run_fuse(model_path, adp_dir, fused_path, best_iter)
    else:
        print("⏭  Fusion skipped (--skip-fuse)")

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()