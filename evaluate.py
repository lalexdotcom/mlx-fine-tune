# evaluate.py
#
# Evaluation pipeline for fine-tuned MLX language models.
# Downloads and caches the evaluation dataset, runs inference using the
# mlx_lm Python API (model stays loaded between examples for speed),
# and produces a structured report with accuracy metrics.

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from lib.utils import short_hash, format_bytes, count_jsonl_lines
from lib.dataset import (
    download_parquet_files,
    parquet_to_raw_jsonl,
    raw_jsonl_path,
    resolve_dataset_format,
)
from lib.model import resolve_model_path, read_tokenizer_config
from formats.registry import load_format

# ─── Paths derived from --work-dir ───────────────────────────────────────────

VENV_PYTHON = None
CACHE_DIR   = None
WORK_DIR    = None
EVALS_DIR   = None


def init_dirs(work_dir: str) -> None:
    """Initialize global path variables from the work directory."""
    global VENV_PYTHON, CACHE_DIR, WORK_DIR, EVALS_DIR
    base        = Path(work_dir)
    VENV_PYTHON = base / "venv" / "bin" / "python3"
    CACHE_DIR   = base / "cache"
    WORK_DIR    = base / "work"
    EVALS_DIR   = base / "evals"

# ─── Inference config ─────────────────────────────────────────────────────────

GENERATE_CONFIG = {
    "max_tokens": 512,
    "temperature": 0.0,   # greedy decoding for deterministic evaluation
    "verbose": False,
}

# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MLX LLM evaluation pipeline")
    p.add_argument("--work-dir", required=True,
                   help="Base working directory (set by evaluate.sh)")
    p.add_argument("--model", "-m", required=True,
                   help="Model path (relative to ~/.lmstudio/models/, absolute, ~/... or ./local)")
    p.add_argument("--dataset", "-d", required=True,
                   help="HuggingFace dataset ID to evaluate against")
    p.add_argument("--dataset-format", default=None,
                   help="Dataset format converter. Auto-detected from cache if previously used.")
    p.add_argument("--run-id", type=str, default="manual",
                   help="Run ID for logging (injected by evaluate.sh)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help=f"Max tokens to generate per example (default: {GENERATE_CONFIG['max_tokens']})")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit evaluation to first N examples (default: all)")
    return p.parse_args()


def apply_overrides(args) -> None:
    """Apply CLI overrides to the global GENERATE_CONFIG dict."""
    if args.max_tokens is not None:
        GENERATE_CONFIG["max_tokens"] = args.max_tokens

# ─── Eval cache helpers ───────────────────────────────────────────────────────

def eval_output_dir(model_name: str, dataset_hash: str, run_id: str) -> Path:
    """Return the output directory for an evaluation run.
    Keyed on model name + dataset hash + run ID (timestamp).
    """
    return EVALS_DIR / f"{model_name}_{dataset_hash}_{run_id}"

# ─── Scoring ──────────────────────────────────────────────────────────────────

def parse_tool_call(text: str) -> dict | None:
    """Attempt to extract a tool call from model output.
    Looks for <tool_call>...</tool_call> XML tags as used by Qwen3 chat template,
    then falls back to raw JSON parsing.
    Returns {"name": ..., "arguments": ...} or None if no tool call found.
    """
    # Try XML tag format first (Qwen3 / ChatML)
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return {
                "name": data.get("name", ""),
                "arguments": data.get("arguments", {}),
            }
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse the whole output as JSON
    try:
        data = json.loads(text.strip())
        if "name" in data:
            return {
                "name": data.get("name", ""),
                "arguments": data.get("arguments", {}),
            }
    except json.JSONDecodeError:
        pass

    return None


def normalize_arguments(args) -> dict:
    """Normalize tool call arguments to a dict for comparison.
    Handles both dict and JSON string formats.
    """
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args or {}


def score_example(
    predicted_text: str,
    expected: "EvalExample",  # noqa: F821
) -> dict:
    """Score a single prediction against the expected output.
    Returns a dict with individual metric scores and details.

    Metrics:
    - tool_name_match: correct tool was called (or neither expected nor predicted)
    - args_match: all expected arguments match (only when tool_name_match is True)
    - false_positive: tool called when none was expected
    - false_negative: no tool called when one was expected
    - correct: overall correctness (tool_name_match + args_match when applicable)
    """
    predicted_tool = parse_tool_call(predicted_text)

    expected_tool = expected.expected_tool
    expected_has_tool = expected_tool is not None
    predicted_has_tool = predicted_tool is not None

    # Detect false positives and false negatives
    false_positive = predicted_has_tool and not expected_has_tool
    false_negative = not predicted_has_tool and expected_has_tool

    tool_name_match = False
    args_match = False

    if not expected_has_tool and not predicted_has_tool:
        # No tool expected, no tool predicted — correct
        tool_name_match = True
        args_match = True
    elif expected_has_tool and predicted_has_tool:
        tool_name_match = (
            predicted_tool["name"].lower() == expected_tool["name"].lower()
        )
        if tool_name_match:
            pred_args = normalize_arguments(predicted_tool.get("arguments", {}))
            exp_args = normalize_arguments(expected_tool.get("arguments", {}))
            # Check all expected keys are present and match
            args_match = all(
                str(pred_args.get(k, "")).lower() == str(v).lower()
                for k, v in exp_args.items()
            )

    correct = tool_name_match and args_match and not false_positive

    return {
        "tool_name_match": tool_name_match,
        "args_match": args_match,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "correct": correct,
        "predicted_tool": predicted_tool,
        "predicted_text": predicted_text,
    }

# ─── Report generation ────────────────────────────────────────────────────────

def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate metrics from a list of scored results."""
    total = len(results)
    if total == 0:
        return {}

    correct        = sum(1 for r in results if r["score"]["correct"])
    tool_match     = sum(1 for r in results if r["score"]["tool_name_match"])
    args_match     = sum(1 for r in results if r["score"]["args_match"])
    false_pos      = sum(1 for r in results if r["score"]["false_positive"])
    false_neg      = sum(1 for r in results if r["score"]["false_negative"])

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "tool_name_match": tool_match,
        "tool_name_accuracy": round(tool_match / total, 4),
        "args_match": args_match,
        "false_positives": false_pos,
        "false_negatives": false_neg,
        "false_positive_rate": round(false_pos / total, 4),
        "false_negative_rate": round(false_neg / total, 4),
    }


def write_report(output_dir: Path, results: list[dict], summary: dict, meta: dict) -> None:
    """Write evaluation results to output_dir in three formats:
    - results.json: all examples with predictions and scores
    - summary.json: aggregated metrics
    - report.md: human-readable markdown report
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # results.json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # summary.json
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"meta": meta, "summary": summary}, f, indent=2)

    # report.md
    with open(output_dir / "report.md", "w") as f:
        f.write(f"# Evaluation Report\n\n")
        f.write(f"**Model**: {meta['model']}\n")
        f.write(f"**Dataset**: {meta['dataset']}\n")
        f.write(f"**Format**: {meta['format']}\n")
        f.write(f"**Date**: {meta['date']}\n")
        f.write(f"**Run ID**: {meta['run_id']}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        for k, v in summary.items():
            if isinstance(v, float):
                f.write(f"| {k} | {v:.1%} |\n")
            else:
                f.write(f"| {k} | {v} |\n")
        f.write(f"\n## Failures\n\n")

        # List incorrect examples
        failures = [r for r in results if not r["score"]["correct"]]
        if not failures:
            f.write("No failures — perfect score! 🎉\n")
        else:
            for i, r in enumerate(failures[:50]):  # cap at 50 in report
                f.write(f"### Example {r['index']}\n")
                f.write(f"**Input**: {r.get('input', '')[:200]}\n\n")
                f.write(f"**Expected tool**: `{r.get('expected_tool')}`\n\n")
                f.write(f"**Predicted tool**: `{r['score']['predicted_tool']}`\n\n")
                f.write(f"**Predicted text**: {r['score']['predicted_text'][:300]}\n\n")
                f.write("---\n\n")
            if len(failures) > 50:
                f.write(f"*{len(failures) - 50} more failures not shown — see results.json*\n")

    print(f"  ✓ results.json")
    print(f"  ✓ summary.json")
    print(f"  ✓ report.md")

# ─── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(
    model_path: Path,
    tokenizer_config: dict,
    fmt_module,
    dataset_id: str,
    dataset_hash: str,
    dataset_format: str,
    output_dir: Path,
    limit: int | None,
    run_id: str,
) -> None:
    """Run the full evaluation loop:
    1. Load the model into memory once via mlx_lm Python API
    2. Stream eval examples from raw JSONL
    3. Generate predictions
    4. Score each prediction
    5. Write report
    """
    from mlx_lm import load, generate

    print(f"\n── Loading model ─────────────────────────────")
    print(f"   {model_path}")
    model, tokenizer = load(str(model_path))
    print(f"   ✓ Model loaded")

    raw_path = raw_jsonl_path(CACHE_DIR, dataset_hash)
    total_lines = count_jsonl_lines(raw_path)
    total = min(total_lines, limit) if limit else total_lines
    print(f"\n── Evaluating {total} examples ────────────────")

    results = []
    index = 0
    start_time = time.time()

    with open(raw_path) as f:
        for line in f:
            if limit and index >= limit:
                break
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                index += 1
                continue

            # Convert example to eval format
            eval_ex = fmt_module.convert_for_eval(example, tokenizer_config)
            if eval_ex is None:
                index += 1
                continue

            # Generate prediction — model stays loaded between calls
            try:
                predicted = generate(
                    model,
                    tokenizer,
                    prompt=eval_ex.prompt,
                    max_tokens=GENERATE_CONFIG["max_tokens"],
                    temp=GENERATE_CONFIG["temperature"],
                    verbose=GENERATE_CONFIG["verbose"],
                )
            except Exception as e:
                predicted = ""
                print(f"\n  ⚠ Generation failed for example {index}: {e}")

            # Score the prediction
            score = score_example(predicted, eval_ex)

            results.append({
                "index": index,
                "input": line[:500],
                "expected_tool": eval_ex.expected_tool,
                "expected_text": eval_ex.expected_text,
                "score": score,
            })

            index += 1

            # Progress
            elapsed = time.time() - start_time
            avg = elapsed / index if index > 0 else 0
            eta = avg * (total - index)
            correct_so_far = sum(1 for r in results if r["score"]["correct"])
            acc = correct_so_far / len(results) if results else 0
            print(
                f"\r  [{index}/{total}] acc: {acc:.1%} "
                f"elapsed: {elapsed:.0f}s eta: {eta:.0f}s   ",
                end="", flush=True,
            )

    print()

    # Compute and write report
    summary = compute_summary(results)
    meta = {
        "model": model_path.name,
        "dataset": dataset_id,
        "format": dataset_format,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "total_examples": total,
        "generate_config": GENERATE_CONFIG,
    }

    print(f"\n── Writing report ─────────────────────────────")
    print(f"   {output_dir}")
    write_report(output_dir, results, summary, meta)

    # Print summary to stdout
    print(f"\n── Results ────────────────────────────────────")
    print(f"   Accuracy        : {summary.get('accuracy', 0):.1%} ({summary.get('correct', 0)}/{summary.get('total', 0)})")
    print(f"   Tool name match : {summary.get('tool_name_accuracy', 0):.1%}")
    print(f"   False positives : {summary.get('false_positives', 0)}")
    print(f"   False negatives : {summary.get('false_negatives', 0)}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    init_dirs(args.work_dir)
    apply_overrides(args)

    print(f"\n── Eval Run {args.run_id} {'─' * 26}")

    # 1. Resolve model path
    model_path = resolve_model_path(args.model)
    print(f"\n── Model ─────────────────────────────────")
    print(f"   {model_path}")

    tokenizer_config = read_tokenizer_config(model_path)
    d_hash = short_hash(args.dataset)

    # 2. Resolve dataset format
    try:
        dataset_format = resolve_dataset_format(
            CACHE_DIR, d_hash, args.dataset_format
        )
        fmt_module = load_format(dataset_format)
    except ValueError as e:
        print(f"\n❌ {e}")
        exit(1)

    print(f"   Dataset          : {args.dataset}")
    print(f"   Dataset hash     : {d_hash}")
    print(f"   Dataset format   : {dataset_format}")

    # 3. Download parquet files
    print(f"\n── Dataset ───────────────────────────────")
    local_paths = download_parquet_files(args.dataset, d_hash, CACHE_DIR)

    # 4. Parquet → raw JSONL (shared cache with training pipeline)
    parquet_to_raw_jsonl(CACHE_DIR, d_hash, local_paths)

    # 5. Run evaluation
    output_dir = eval_output_dir(model_path.name, d_hash, args.run_id)

    run_evaluation(
        model_path=model_path,
        tokenizer_config=tokenizer_config,
        fmt_module=fmt_module,
        dataset_id=args.dataset,
        dataset_hash=d_hash,
        dataset_format=dataset_format,
        output_dir=output_dir,
        limit=args.limit,
        run_id=args.run_id,
    )

    print(f"\n✅ Evaluation complete.")
    print(f"   Report: {output_dir}")


if __name__ == "__main__":
    main()