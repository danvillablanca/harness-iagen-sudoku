"""Generate Sudoku task splits (train/eval/test) with difficulty levels.

Adapted from the Sudoku generator in Reasoning Gym (Stojanovski et al., 2025):
a random solved grid is created, then cells are removed one at a time,
optionally keeping a removal only while the solution stays unique.
Unlike Reasoning Gym, it supports 4x4, 6x6 and 9x9 boards, records the
actual number of blank cells, and keeps puzzles disjoint across splits.

Every record is generated from its own seed, so the same arguments always
produce the same files. Run from the repository root:

    python Laboratorio/Qwen3.5/dataset.py --out data/sudoku
"""

import argparse
import json
import random
from pathlib import Path

from prompts import PromptCase
from sudoku import (
    build_prompt,
    count_solutions,
    is_deducible,
    load_sudoku,
    random_solution,
)

TEMPLATE = load_sudoku(Path(__file__).with_name("sudoku.json"))["prompt_template"]

BOARDS = {4: (2, 2), 6: (2, 3), 9: (3, 3)}

# Inclusive ranges of blank cells per level. Not yet calibrated against the
# target model; adjust after measuring pass@k per level.
LEVELS = {
    4: {"easy": (4, 6), "medium": (7, 9), "hard": (10, 12)},
    6: {"easy": (8, 12), "medium": (13, 18), "hard": (19, 24)},
    9: {"easy": (10, 20), "medium": (21, 35), "hard": (36, 50)},
}

SPLITS = ("train", "eval", "test")

MAX_ATTEMPTS = 200


def board(size: int) -> dict:
    box_rows, box_columns = BOARDS[size]
    return {
        "size": size,
        "box_rows": box_rows,
        "box_columns": box_columns,
        "prompt_template": TEMPLATE,
    }


def blank_cells(solution, case, empty_cells, rng, unique=True):
    """Remove `empty_cells` random cells, or return None if it cannot.

    With `unique`, a removal is undone when it would allow a second solution.
    """
    puzzle = [row[:] for row in solution]
    size = case["size"]
    positions = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(positions)
    removed = 0
    for r, c in positions:
        if removed == empty_cells:
            break
        value = puzzle[r][c]
        puzzle[r][c] = 0
        if unique and count_solutions(puzzle, case) != 1:
            puzzle[r][c] = value
            continue
        removed += 1
    return puzzle if removed == empty_cells else None


def make_record(split, size, level, index, seed=0, unique=True, seen=None):
    """One task, deterministic in (seed, split, size, level, index).

    Puzzles already in `seen` are skipped, so splits never share a puzzle.
    """
    case = board(size)
    low, high = LEVELS[size][level]
    for attempt in range(MAX_ATTEMPTS):
        rng = random.Random(f"{seed}:{split}:{size}:{level}:{index}:{attempt}")
        empty_cells = rng.randint(low, high)
        solution = random_solution(case, rng.randrange(2**32))
        puzzle = blank_cells(solution, case, empty_cells, rng, unique)
        if puzzle is None:
            continue
        key = (size, json.dumps(puzzle))
        if seen is not None:
            if key in seen:
                continue
            seen.add(key)
        data = {**case, "puzzle": puzzle}
        return {
            "id": f"{split}-{size}x{size}-{level}-{index:05d}",
            "split": split,
            "size": size,
            "box_rows": case["box_rows"],
            "box_columns": case["box_columns"],
            "level": level,
            "empty_cells": sum(value == 0 for row in puzzle for value in row),
            "unique_solution": count_solutions(puzzle, case) == 1,
            "deducible_by_singles": is_deducible(puzzle, case),
            "prompt": build_prompt(data),
            "puzzle": puzzle,
            "solution": solution,
        }
    raise ValueError(
        f"Could not generate {split} {size}x{size} {level} #{index} "
        f"after {MAX_ATTEMPTS} attempts."
    )


def generate(counts, sizes=(4, 6, 9), seed=0, unique=True):
    """`counts` maps each split to the number of tasks per size and level.

    Splits are generated in order (test first), so adding training tasks
    never changes the evaluation and test sets.
    """
    seen = set()
    dataset = {split: [] for split in SPLITS}
    for split in ("test", "eval", "train"):
        for size in sizes:
            for level in LEVELS[size]:
                for index in range(counts.get(split, 0)):
                    dataset[split].append(
                        make_record(split, size, level, index, seed, unique, seen)
                    )
    return dataset


def save(dataset, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, records in dataset.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")


def load(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def case_from_record(record) -> PromptCase:
    """PromptCase usable with `evaluate`, `print_result` and the model."""
    data = {
        "name": record["id"],
        "size": record["size"],
        "box_rows": record["box_rows"],
        "box_columns": record["box_columns"],
        "puzzle": record["puzzle"],
        "expected_solution": record["solution"],
        "prompt_template": TEMPLATE,
    }
    return PromptCase(prompt=record["prompt"], data=data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Sudoku task splits.")
    parser.add_argument("--out", default="data/sudoku", help="output directory")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 9],
                        choices=sorted(BOARDS))
    parser.add_argument("--train", type=int, default=50,
                        help="training tasks per size and level")
    parser.add_argument("--eval", type=int, default=10,
                        help="evaluation tasks per size and level")
    parser.add_argument("--test", type=int, default=10,
                        help="test tasks per size and level")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--unique",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require a unique solution (default: required)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = {"train": args.train, "eval": args.eval, "test": args.test}
    if any(count < 0 for count in counts.values()):
        raise SystemExit("Task counts must be >= 0.")
    dataset = generate(counts, args.sizes, args.seed, args.unique)
    save(dataset, args.out)
    for split, records in dataset.items():
        print(f"{split}: {len(records)} tasks -> {Path(args.out) / f'{split}.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
