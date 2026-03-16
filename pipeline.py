import argparse
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
import urllib.parse
import yaml
from pathlib import Path

import pyarrow.parquet as pq
from jinja2 import Environment, BaseLoader

# ─── Paths are all derived from --work-dir, passed by mlx-fine-tune.sh ────────

VENV_PYTHON    = None
CACHE_DIR      = None
WORK_DIR       = None

def init_dirs(work_dir: str) -> None:
    global VENV_PYTHON, CACHE_DIR, WORK_DIR
    base        = Path(work_dir)
    VENV_PYTHON = base / "venv" / "bin" / "python3"
    CACHE_DIR   = base / "cache"
    WORK_DIR    = base / "work"

# ─── Constants ────────────────────────────────────────────────────────────────

LM_STUDIO_MODELS_DIR = (
    Path.home() / ".lmstudio" / "models"
    if (Path.home() / ".lmstudio" / "models").exists()
    else Path.home() / ".cache" / "lm-studio" / "models"
)

# ─── Fine-tuning config ───────────────────────────────────────────────────────

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

# ─── Utils ────────────────────────────────────────────────────────────────────

def format_bytes(b: int) -> str:
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / 1024 / 1024:.1f} MB"


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def get_token_string(token) -> str:
    if token is None:
        return ""
    if isinstance(token, str):
        return token
    if isinstance(token, dict):
        return token.get("content", "")
    return ""


def build_fused_path(model_path: Path, fused_base: str, name: str) -> Path:
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


def hf_parquet_api(dataset_id: str) -> str:
    return (
        "https://datasets-server.huggingface.co/parquet"
        f"?dataset={urllib.parse.quote(dataset_id, safe='')}"
    )

# ─── Cache key helpers ────────────────────────────────────────────────────────

def parquet_cache_dir(dataset_hash: str) -> Path:
    return CACHE_DIR / "parquet" / dataset_hash

def raw_jsonl_path(dataset_hash: str) -> Path:
    return CACHE_DIR / "raw" / dataset_hash / "rows.jsonl"

def template_cache_dir(dataset_hash: str, template_hash: str) -> Path:
    return CACHE_DIR / "template" / f"{dataset_hash}_{template_hash}"

def adapters_dir(model_name: str, dataset_hash: str, template_hash: str) -> Path:
    return WORK_DIR / "adapters" / f"{model_name}_{dataset_hash}_{template_hash}"

def data_dir(dataset_hash: str, template_hash: str) -> Path:
    return WORK_DIR / "data" / f"{dataset_hash}_{template_hash}"

# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MLX LLM fine-tuning pipeline")
    p.add_argument("--work-dir", required=True,
                   help="Base working directory (set by mlx-fine-tune.sh)")
    p.add_argument("--dataset", "-d", required=True,
                   help="HuggingFace dataset id (e.g. acon96/Home-Assistant-Requests-V2)")
    p.add_argument("--path", "-p", required=True,
                   help="Model path (relative to LM Studio, absolute, ~/... or ./local)")
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
                   help="Run ID for logging (injected by mlx-fine-tune.sh)")
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


def apply_overrides(args):
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

# ─── Model path resolution ────────────────────────────────────────────────────

def resolve_model_path(input_path: str) -> Path:
    p = input_path
    if p.startswith("~/"):
        p = str(Path.home() / p[2:])
    path = Path(p)
    if path.is_absolute():
        return path
    if p.startswith("./") or p.startswith("../"):
        return Path.cwd() / path
    return LM_STUDIO_MODELS_DIR / path

# ─── Tokenizer ────────────────────────────────────────────────────────────────

def read_tokenizer_config(model_path: Path) -> dict:
    config_path = model_path / "tokenizer_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"tokenizer_config.json not found in: {model_path}")
    with open(config_path) as f:
        config = json.load(f)
    if not config.get("chat_template"):
        raise ValueError(f"No chat_template found in {config_path}")
    print("✓ tokenizer_config.json loaded")
    return config

# ─── Parquet download ─────────────────────────────────────────────────────────

def fetch_parquet_urls(dataset_id: str) -> list[dict]:
    with urllib.request.urlopen(hf_parquet_api(dataset_id)) as r:
        data = json.loads(r.read())
    return [f for f in data["parquet_files"] if f["split"] == "train"]


def download_file(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while chunk := r.read(1024 * 1024):
            f.write(chunk)


def download_parquet_files(dataset_id: str, dataset_hash: str) -> list[Path]:
    cache_dir = parquet_cache_dir(dataset_hash)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching parquet file list from HuggingFace...")
    files = fetch_parquet_urls(dataset_id)
    print(f"  {len(files)} parquet file(s) to download")

    local_paths = []

    for i, file in enumerate(files):
        local_path = cache_dir / file["filename"]
        local_paths.append(local_path)

        if local_path.exists():
            local_size = local_path.stat().st_size
            if local_size == file["size"]:
                print(f"  [{i+1}/{len(files)}] {file['filename']} — already cached ({format_bytes(file['size'])})")
                continue
            print(f"  [{i+1}/{len(files)}] {file['filename']} — incomplete ({format_bytes(local_size)}/{format_bytes(file['size'])}) — re-downloading")

        print(f"  [{i+1}/{len(files)}] Downloading {file['filename']} ({format_bytes(file['size'])})...", end="", flush=True)
        download_file(file["url"], local_path)
        print(" ✓")

    return local_paths

# ─── Raw JSONL cache ──────────────────────────────────────────────────────────

def count_parquet_rows(local_paths: list[Path]) -> int:
    total = 0
    for path in local_paths:
        pf = pq.ParquetFile(path)
        total += pf.metadata.num_rows
    return total


def raw_cache_valid(dataset_hash: str, local_paths: list[Path]) -> bool:
    raw_path = raw_jsonl_path(dataset_hash)
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return False
    parquet_total = count_parquet_rows(local_paths)
    jsonl_lines = count_jsonl_lines(raw_path)
    if jsonl_lines != parquet_total:
        print(f"  ⚠ Raw JSONL incomplete ({jsonl_lines}/{parquet_total} rows) — re-converting")
        return False
    return True

# ─── Template cache ───────────────────────────────────────────────────────────

def template_cache_valid(dataset_hash: str, tmpl_hash: str) -> bool:
    d = template_cache_dir(dataset_hash, tmpl_hash)
    for split in ["train", "valid", "test"]:
        path = d / f"{split}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            return False
        if count_jsonl_lines(path) == 0:
            return False
    return True


def copy_template_cache(dataset_hash: str, tmpl_hash: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = template_cache_dir(dataset_hash, tmpl_hash)
    for split in ["train", "valid", "test"]:
        src = src_dir / f"{split}.jsonl"
        dst = out_dir / f"{split}.jsonl"
        shutil.copyfile(src, dst)
        lines = count_jsonl_lines(dst)
        print(f"  {dst} — {lines} examples (from cache)")

# ─── Step 1: Parquet → raw JSONL ─────────────────────────────────────────────

def parquet_to_raw_jsonl(dataset_hash: str, local_paths: list[Path]) -> None:
    raw_path = raw_jsonl_path(dataset_hash)

    if raw_cache_valid(dataset_hash, local_paths):
        print(f"✓ Raw JSONL cache valid ({format_bytes(raw_path.stat().st_size)}) — skipping parquet conversion")
        return

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    total = count_parquet_rows(local_paths)
    print(f"Converting parquet → raw JSONL...")
    print(f"  {total} total rows across {len(local_paths)} file(s)")

    written = 0
    with open(raw_path, "w") as out:
        for fi, path in enumerate(local_paths):
            pf = pq.ParquetFile(path)
            num_groups = pf.metadata.num_row_groups
            print(f"  File [{fi+1}/{len(local_paths)}]: {path.name} — {num_groups} row group(s)", flush=True)

            for rg in range(num_groups):
                table = pf.read_row_group(rg)
                rows = table.to_pylist()
                del table

                for row in rows:
                    out.write(json.dumps(row) + "\n")
                    written += 1

                pct = int(written / total * 100)
                print(f"\r  [{pct:3d}%] {written}/{total} rows written   ", end="", flush=True)

    print(flush=True)
    print(f"  ✓ Raw JSONL written to {raw_path}")

# ─── Step 2: raw JSONL → masked JSONL ────────────────────────────────────────

def render_turn(env, template_str, eos_token, bos_token, messages_slice, tools):
    tmpl = env.from_string(template_str)
    return tmpl.render(
        messages=messages_slice,
        tools=tools if tools else None,
        eos_token=eos_token,
        bos_token=bos_token,
        add_generation_prompt=False,
    )


def convert_example(env, template_str, eos_token, bos_token, example):
    messages = example.get("messages") or []
    tools = example.get("tools") or None
    results = []
    previous_rendered = ""

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        is_assistant = role in ("assistant", "model")
        should_train = bool(msg.get("train_on_turn", False))

        messages_slice = []
        for m in messages[: i + 1]:
            content = m.get("content", "")
            if isinstance(content, list):
                content = content[0].get("text", "") if content else ""
            entry = {"role": m["role"], "content": content}
            if m.get("tool_calls"):
                entry["tool_calls"] = m["tool_calls"]
            messages_slice.append(entry)

        try:
            current_rendered = render_turn(
                env, template_str, eos_token, bos_token, messages_slice, tools
            )
        except Exception:
            continue

        turn_text = current_rendered[len(previous_rendered):]
        previous_rendered = current_rendered

        if turn_text and is_assistant and should_train:
            prompt = previous_rendered[: len(previous_rendered) - len(turn_text)]
            results.append({"text": prompt + turn_text})

    return results


def assign_split(idx: int, total: int) -> str:
    r = idx / total
    if r < 0.8:
        return "train"
    if r < 0.9:
        return "valid"
    return "test"


def raw_jsonl_to_masked(
    dataset_hash: str,
    tokenizer_config: dict,
    tmpl_hash: str,
) -> None:
    raw_path = raw_jsonl_path(dataset_hash)
    out_dir = template_cache_dir(dataset_hash, tmpl_hash)

    template_str = tokenizer_config["chat_template"]
    eos_token = get_token_string(tokenizer_config.get("eos_token"))
    bos_token = get_token_string(tokenizer_config.get("bos_token"))

    env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
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

            masked = convert_example(env, template_str, eos_token, bos_token, example)
            split = assign_split(index, total)
            index += 1

            for item in masked:
                writers[split].write(json.dumps(item) + "\n")
                counts[split] += 1

            if index % 1000 == 0:
                pct = int(index / total * 100)
                print(
                    f"\r  [{pct:3d}%] {index}/{total} — "
                    f"train: {counts['train']} "
                    f"valid: {counts['valid']} "
                    f"test: {counts['test']}   ",
                    end="",
                    flush=True,
                )

    print(flush=True)
    for name, f in writers.items():
        f.close()
        print(f"  {out_dir}/{name}.jsonl — {counts[name]} examples")

# ─── Run command ──────────────────────────────────────────────────────────────

def run_command(cmd: list[str]) -> None:
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

    # Check for existing checkpoint to resume from
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

    # Save best iter for future use
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

    # 1. Compute cache keys
    d_hash = short_hash(args.dataset)

    # 2. Resolve model path and compute keys
    model_path = resolve_model_path(args.path)
    print(f"\n── Model ─────────────────────────────────")
    print(f"   {model_path}")

    tokenizer_config = read_tokenizer_config(model_path)
    t_hash = short_hash(tokenizer_config["chat_template"])

    print(f"   Dataset hash       : {d_hash}  ({args.dataset})")
    print(f"   Template hash      : {t_hash}")

    fused_path = build_fused_path(model_path, args.fused_base, args.name)
    print(f"   Fused model target : {fused_path}")

    # 3. Download parquet files
    print(f"\n── Dataset ───────────────────────────────")
    local_paths = download_parquet_files(args.dataset, d_hash)

    # 4. Parquet → raw JSONL (cached per dataset)
    parquet_to_raw_jsonl(d_hash, local_paths)

    # 5. Raw JSONL → masked JSONL (cached per dataset + template)
    out_data_dir = data_dir(d_hash, t_hash)

    if template_cache_valid(d_hash, t_hash):
        print(f"✓ Template cache hit ({d_hash}_{t_hash}) — skipping conversion")
        copy_template_cache(d_hash, t_hash, out_data_dir)
    else:
        print(f"\nApplying chat template (Jinja2)...")
        raw_jsonl_to_masked(d_hash, tokenizer_config, t_hash)
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