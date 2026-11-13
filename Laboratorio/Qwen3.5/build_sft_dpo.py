"""Build SFT and DPO datasets from the traces written by `baseline.py run`.

Reads the sample files of a teacher run (files or directories, e.g.
`results/traces/r0`), recovers each prompt from the tasks JSONL by `task_id` (the
shots used for generation are not part of the training prompt), and writes:

- `sft.jsonl`: `prompt`, `completion` from fully correct, non-truncated samples,
  the shortest `--max-sft-per-task` distinct ones per task.
- `dpo.jsonl`: `prompt`, `chosen`, `rejected` pairs per task, at most
  `--max-pairs-per-task`, each completion used once. Chosen is a correct, non-truncated
  sample (shortest first). Rejected prefers wrong grids that finished (closest to correct
  first) over truncated ones, because truncation mostly teaches "be shorter".

Run from the repository root:

    python Laboratorio/Qwen3.5/build_sft_dpo.py --results results/traces/r0 \\
        --tasks data/sudoku/train.jsonl --out data/traces
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from baseline import load_samples
from dataset import load


def sample_files(paths):
    """Expand directories to the `*.jsonl` files below them."""
    files = []
    for path in map(Path, paths):
        files.extend(sorted(path.rglob("*.jsonl")) if path.is_dir() else [path])
    return files


def group_by_task(samples):
    """{task_id: [samples]}; refuses samples that are not from the train split."""
    groups = defaultdict(list)
    for sample in samples:
        if sample.get("split") != "train":
            raise ValueError(
                f"{sample['task_id']} is from split {sample.get('split')!r}; "
                "only train traces may be used for training data."
            )
        groups[sample["task_id"]].append(sample)
    return groups


def is_correct(sample):
    """Fully correct and not cut off (a truncated sample cannot be correct anyway)."""
    return sample["reward"] == 1.0 and sample["finish_reason"] == "stop"


def shortest_distinct(samples, limit):
    """Up to `limit` samples with different completions, fewest tokens first."""
    chosen, seen = [], set()
    for sample in sorted(samples, key=lambda s: (s["generated_tokens"], s["sample"])):
        if sample["completion"] not in seen:
            seen.add(sample["completion"])
            chosen.append(sample)
        if len(chosen) == limit:
            break
    return chosen


def reject_order(sample):
    """Wrong grids that finished first, closest to correct first; then truncated."""
    finished = sample["finish_reason"] == "stop"
    return (0 if finished else 1, -sample["reward"], sample["sample"])


def sft_examples(groups, prompts, max_per_task):
    examples = []
    for task_id in sorted(groups):
        correct = [s for s in groups[task_id] if is_correct(s)]
        for sample in shortest_distinct(correct, max_per_task):
            examples.append({
                "task_id": task_id, "size": sample["size"], "level": sample["level"],
                "prompt": prompts[task_id], "completion": sample["completion"],
            })
    return examples


def dpo_pairs(groups, prompts, max_per_task):
    pairs = []
    for task_id in sorted(groups):
        samples = groups[task_id]
        chosen = shortest_distinct([s for s in samples if is_correct(s)], max_per_task)
        rejected = sorted((s for s in samples if s["reward"] < 1.0), key=reject_order)
        for good, bad in zip(chosen, rejected):
            pairs.append({
                "task_id": task_id, "size": good["size"], "level": good["level"],
                "prompt": prompts[task_id], "chosen": good["completion"],
                "rejected": bad["completion"],
                "rejected_truncated": bad["finish_reason"] != "stop",
            })
    return pairs


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def describe(name, rows, extra=None):
    counts = Counter((row["size"], row["level"]) for row in rows)
    detail = ", ".join(f"{size}x{size} {level}: {n}"
                       for (size, level), n in sorted(counts.items()))
    print(f"{name}: {len(rows)} ({detail})")
    if extra:
        print(extra)


def parse_args():
    parser = argparse.ArgumentParser(description="Build SFT and DPO data from traces.")
    parser.add_argument("--results", nargs="+", required=True,
                        help="sample JSONL files or directories from baseline.py run")
    parser.add_argument("--tasks", default="data/sudoku/train.jsonl")
    parser.add_argument("--out", default="data/traces", help="output directory")
    parser.add_argument("--max-sft-per-task", type=int, default=2)
    parser.add_argument("--max-pairs-per-task", type=int, default=2)
    args = parser.parse_args()
    if args.max_sft_per_task < 1 or args.max_pairs_per_task < 1:
        parser.error("per-task limits must be positive.")
    return args


def main() -> int:
    args = parse_args()
    prompts = {record["id"]: record["prompt"] for record in load(args.tasks)}
    samples = [s for path in sample_files(args.results) for s in load_samples(path)]
    groups = group_by_task(samples)
    missing = sorted(set(groups) - set(prompts))
    if missing:
        raise ValueError(f"Tasks not in {args.tasks}: {', '.join(missing[:3])} ...")

    sft = sft_examples(groups, prompts, args.max_sft_per_task)
    dpo = dpo_pairs(groups, prompts, args.max_pairs_per_task)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "sft.jsonl", sft)
    write_jsonl(out / "dpo.jsonl", dpo)

    print(f"{len(samples)} samples, {len(groups)} tasks -> {out}/")
    describe("sft", sft)
    truncated = sum(row["rejected_truncated"] for row in dpo)
    describe("dpo", dpo, f"  rejected truncated: {truncated}, wrong grid: {len(dpo) - truncated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
