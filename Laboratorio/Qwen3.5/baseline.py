"""Baseline: pass@1 and pass@k per model, board size and difficulty level.

`run` samples `n` completions per task with one model, scores each with
`reward`, and appends one JSON line per sample. Rerunning with the same
arguments resumes where it stopped (useful when Colab disconnects).
`summary` aggregates one or more of those files into a table.
Run from the repository root:

    python Laboratorio/Qwen3.5/dataset.py --out data/sudoku
    python Laboratorio/Qwen3.5/baseline.py run --tasks data/sudoku/eval.jsonl \\
        --model-id Qwen/Qwen3.5-2B --no-thinking --per-level 5 --n 8
    python Laboratorio/Qwen3.5/baseline.py summary results/*.jsonl
"""

import argparse
import json
import zlib
from collections import defaultdict
from math import comb
from pathlib import Path

from dataset import load
from reward import reward
from timing import timed

# Settings recorded with every sample; resuming requires the same values.
RUN_KEYS = ("model", "thinking", "temperature", "top_p", "max_tokens", "n", "seed")

# Sampling defaults recommended by Qwen for each mode.
SAMPLING = {True: (0.6, 0.95), False: (0.7, 0.8)}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Requires 0 <= c <= n and 1 <= k <= n.")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def select_tasks(records, per_level=None, sizes=None):
    """First `per_level` tasks of each (size, level), in file order."""
    taken = defaultdict(int)
    selected = []
    for record in records:
        if sizes and record["size"] not in sizes:
            continue
        key = (record["size"], record["level"])
        if per_level is None or taken[key] < per_level:
            taken[key] += 1
            selected.append(record)
    return selected


def output_path(out_dir, model_id, thinking):
    mode = "thinking" if thinking else "nothinking"
    return Path(out_dir) / f"{model_id.replace('/', '__')}-{mode}.jsonl"


def load_samples(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pending(tasks, done, settings):
    """(task, sample indices still missing), checking settings of earlier runs."""
    finished = defaultdict(set)
    for sample in done:
        if any(sample[key] != settings[key] for key in RUN_KEYS):
            raise ValueError(
                f"Existing results use different settings ({sample['task_id']}); "
                "use another --out directory or the original arguments."
            )
        finished[sample["task_id"]].add(sample["sample"])
    work = []
    for task in tasks:
        missing = [i for i in range(settings["n"]) if i not in finished[task["id"]]]
        if missing:
            work.append((task, missing))
    return work


def task_seed(seed, task_id, first_sample):
    return zlib.crc32(f"{seed}:{task_id}:{first_sample}".encode())


def run(args):
    temperature, top_p = SAMPLING[args.thinking]
    temperature = args.temperature if args.temperature is not None else temperature
    top_p = args.top_p if args.top_p is not None else top_p
    settings = {
        "model": args.model_id,
        "thinking": args.thinking,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": args.max_output_tokens,
        "n": args.n,
        "seed": args.seed,
    }
    tasks = select_tasks(load(args.tasks), args.per_level, args.sizes)
    path = output_path(args.out, args.model_id, args.thinking)
    work = pending(tasks, load_samples(path), settings)
    total = sum(len(missing) for _, missing in work)
    print(f"{len(tasks)} tasks, {total} samples pending -> {path}")
    if not work:
        return 0

    # Imported here so `summary` and the tests do not need PyTorch.
    from config import ModelConfig
    from models import load_model

    config = ModelConfig(
        model_id=args.model_id,
        backend=args.backend,
        thinking=args.thinking,
        max_output_tokens=args.max_output_tokens,
    )
    model, load_seconds = timed(load_model, config)
    print(f"Model loaded in {load_seconds:.1f} s ({model.backend}).")

    path.parent.mkdir(parents=True, exist_ok=True)
    finished = 0
    with path.open("a", encoding="utf-8") as handle:
        for task, missing in work:
            for start in range(0, len(missing), args.batch_size):
                indices = missing[start:start + args.batch_size]
                results, seconds = timed(
                    model.generate_samples,
                    task["prompt"],
                    len(indices),
                    temperature=temperature,
                    top_p=top_p,
                    seed=task_seed(args.seed, task["id"], indices[0]),
                )
                scores = [reward(task["prompt"], result.text) for result in results]
                for index, result, score in zip(indices, results, scores):
                    sample = {
                        **settings,
                        "task_id": task["id"],
                        "split": task["split"],
                        "size": task["size"],
                        "level": task["level"],
                        "empty_cells": task["empty_cells"],
                        "sample": index,
                        "reward": score,
                        "finish_reason": result.finish_reason,
                        "generated_tokens": result.generated_tokens,
                        "seconds": round(seconds / len(indices), 2),
                        "completion": result.text,
                    }
                    handle.write(json.dumps(sample) + "\n")
                handle.flush()
                finished += len(indices)
                print(f"[{finished}/{total}] {task['id']}: "
                      f"{sum(scores):.0f}/{len(scores)} correct ({seconds:.0f} s)")
    return 0


def summarize(samples, ks=(1, 2, 4, 8)):
    """One row per (model, thinking, size, level) with mean pass@k over tasks.

    Tasks with fewer than k samples are left out of that pass@k column.
    """
    tasks = defaultdict(list)
    for sample in samples:
        key = (sample["model"], sample["thinking"], sample["size"], sample["level"])
        tasks[key, sample["task_id"]].append(sample)

    groups = defaultdict(list)
    for (key, _), task_samples in tasks.items():
        groups[key].append(task_samples)

    rows = []
    for (model, thinking, size, level), group in sorted(groups.items()):
        row = {"model": model, "thinking": thinking, "size": size,
               "level": level, "tasks": len(group)}
        for k in ks:
            values = [
                pass_at_k(len(s), sum(x["reward"] == 1.0 for x in s), k)
                for s in group if len(s) >= k
            ]
            row[f"pass@{k}"] = sum(values) / len(values) if values else None
        flat = [x for s in group for x in s]
        row["truncated"] = sum(x["finish_reason"] == "length" for x in flat) / len(flat)
        row["avg_tokens"] = sum(x["generated_tokens"] for x in flat) / len(flat)
        rows.append(row)
    return rows


LEVEL_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def format_table(rows, ks=(1, 2, 4, 8)):
    """Markdown table, ready for the README or the report."""
    rows = sorted(rows, key=lambda r: (r["model"], r["thinking"], r["size"],
                                       LEVEL_ORDER.get(r["level"], 99)))
    header = ["model", "thinking", "board", "level", "tasks",
              *(f"pass@{k}" for k in ks), "truncated", "avg tokens"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["model"], "yes" if row["thinking"] else "no",
                 f"{row['size']}x{row['size']}", row["level"], str(row["tasks"])]
        cells += ["-" if row[f"pass@{k}"] is None else f"{row[f'pass@{k}']:.2f}" for k in ks]
        cells += [f"{row['truncated']:.0%}", f"{row['avg_tokens']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary(args):
    samples = [sample for path in args.files for sample in load_samples(path)]
    if not samples:
        raise SystemExit("No samples found.")
    rows = summarize(samples, args.k)
    print(format_table(rows, args.k))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Sudoku baseline: pass@1 and pass@k.")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="sample and score one model")
    run_parser.add_argument("--tasks", required=True, help="JSONL from dataset.py")
    run_parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    run_parser.add_argument("--backend", default="auto",
                            choices=["auto", "mlx", "cuda", "mps", "cpu"])
    run_parser.add_argument("--thinking", action=argparse.BooleanOptionalAction,
                            default=False, help="Qwen reasoning mode (default: off)")
    run_parser.add_argument("--n", type=int, default=8, help="samples per task")
    run_parser.add_argument("--batch-size", type=int, default=8,
                            help="samples generated together; lower it if the GPU runs out of memory")
    run_parser.add_argument("--max-output-tokens", type=int, default=2048)
    run_parser.add_argument("--temperature", type=float, default=None,
                            help="default: 0.6 with thinking, 0.7 without")
    run_parser.add_argument("--top-p", type=float, default=None,
                            help="default: 0.95 with thinking, 0.8 without")
    run_parser.add_argument("--per-level", type=int, default=None,
                            help="tasks per size and level (default: all)")
    run_parser.add_argument("--sizes", type=int, nargs="+", default=None)
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--out", default="results", help="output directory")

    summary_parser = commands.add_parser("summary", help="aggregate result files")
    summary_parser.add_argument("files", nargs="+")
    summary_parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    summary_parser.add_argument("--out", default=None, help="optional JSON output")

    args = parser.parse_args()
    if args.command == "run" and (args.n < 1 or args.batch_size < 1):
        parser.error("--n and --batch-size must be positive.")
    return args


def main() -> int:
    args = parse_args()
    return run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
