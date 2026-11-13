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
- optional synthetic pairs (`--synthetic-per-task`): the chosen completion with one
  originally blank cell of its final `<answer>` grid changed, so the reject is a finished,
  well-formed but wrong grid. The teacher rarely produces those (its failures are almost
  all truncations). Each is checked with `reward` (format ok, not solved) and marked
  `rejected_synthetic`. The reasoning above the answer is left as written.

Run from the repository root:

    python Laboratorio/Qwen3.5/build_sft_dpo.py --results results/traces/r0 \\
        --tasks data/sudoku/train.jsonl --out data/traces
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from baseline import load_samples
from dataset import load
from reward import reward_parts

ANSWER = re.compile(r"<answer>(.*?)</answer>\s*$", re.DOTALL)


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
                "rejected_synthetic": False,
            })
    return pairs


def corrupt_answer(completion, puzzle, rng):
    """`completion` with one originally blank cell of its final answer changed.

    Returns None when there is no parsable final `<answer>` grid. Changing any digit of a
    completed valid grid breaks its row, column and box.
    """
    match = ANSWER.search(completion)
    if match is None:
        return None
    try:
        grid = json.loads(match.group(1))
    except ValueError:
        return None
    size = len(puzzle)
    blanks = [(r, c) for r in range(size) for c in range(size) if puzzle[r][c] == 0]
    if not blanks or len(grid) != size:
        return None
    row, column = rng.choice(blanks)
    grid[row][column] = rng.choice([d for d in range(1, size + 1) if d != grid[row][column]])
    return completion[:match.start(1)] + json.dumps(grid) + completion[match.end(1):]


def is_wrong_grid(prompt, completion):
    """Well-formed answer that does not solve the task."""
    parts = reward_parts(prompt, completion)
    return parts["format"] == 1.0 and not parts["solved"]


def synthetic_pairs(groups, prompts, puzzles, per_task, used=None):
    """`per_task` pairs per task: a correct completion vs a wrong-grid copy of it.

    `used` maps a task to how many of its correct completions earlier pairs already
    took as `chosen`; the next ones are taken first, repeating only when none are left.
    """
    pairs = []
    for task_id in sorted(groups):
        correct = shortest_distinct(
            [s for s in groups[task_id] if is_correct(s)], len(groups[task_id]))
        offset = (used or {}).get(task_id, 0)
        for k in range(per_task if correct else 0):
            good = correct[(offset + k) % len(correct)]
            bad = corrupt_answer(good["completion"], puzzles[task_id],
                                 random.Random(f"{task_id}:{k}"))
            if bad is None or not is_wrong_grid(prompts[task_id], bad):
                continue
            pairs.append({
                "task_id": task_id, "size": good["size"], "level": good["level"],
                "prompt": prompts[task_id], "chosen": good["completion"],
                "rejected": bad, "rejected_truncated": False, "rejected_synthetic": True,
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
    parser.add_argument("--synthetic-per-task", type=int, default=0,
                        help="extra pairs per task with a corrupted copy as rejected "
                             "(default: 0)")
    args = parser.parse_args()
    if args.max_sft_per_task < 1 or args.max_pairs_per_task < 1:
        parser.error("per-task limits must be positive.")
    if args.synthetic_per_task < 0:
        parser.error("--synthetic-per-task must be >= 0.")
    return args


def main() -> int:
    args = parse_args()
    records = load(args.tasks)
    prompts = {record["id"]: record["prompt"] for record in records}
    puzzles = {record["id"]: record["puzzle"] for record in records}
    samples = [s for path in sample_files(args.results) for s in load_samples(path)]
    groups = group_by_task(samples)
    missing = sorted(set(groups) - set(prompts))
    if missing:
        raise ValueError(f"Tasks not in {args.tasks}: {', '.join(missing[:3])} ...")

    sft = sft_examples(groups, prompts, args.max_sft_per_task)
    dpo = dpo_pairs(groups, prompts, args.max_pairs_per_task)
    used = Counter(pair["task_id"] for pair in dpo)
    dpo += synthetic_pairs(groups, prompts, puzzles, args.synthetic_per_task, used)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "sft.jsonl", sft)
    write_jsonl(out / "dpo.jsonl", dpo)

    print(f"{len(samples)} samples, {len(groups)} tasks -> {out}/")
    describe("sft", sft)
    truncated = sum(row["rejected_truncated"] for row in dpo)
    synthetic = sum(row["rejected_synthetic"] for row in dpo)
    describe("dpo", dpo, f"  rejected truncated: {truncated}, wrong grid: "
                         f"{len(dpo) - truncated - synthetic}, synthetic: {synthetic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
