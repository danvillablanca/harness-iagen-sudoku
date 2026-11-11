"""Named puzzle cases. Add entries as new cases are curated."""

from dataclasses import dataclass
from pathlib import Path

import json

from sudoku import (
    build_prompt,
    cell_candidates,
    load_sudoku,
    random_solution,
    solution_steps,
    with_empty_cells,
)


@dataclass(frozen=True)
class PromptCase:
    prompt: str
    data: dict
    # (user prompt, assistant reply) pairs shown before `prompt`.
    shots: tuple = ()


def _case(filename):
    data = load_sudoku(Path(__file__).with_name(filename))
    return PromptCase(prompt=build_prompt(data), data=data)


# The current puzzle is labeled hard for selection; difficulty has not yet
# been calibrated against the target model.
prompts = {"sudoku9x9": {"hard": _case("sudoku.json")}}


# Blank cells per worked example; each one adds a full-grid snapshot.
SHOT_EMPTY_CELLS = 8


def _grid_lines(grid):
    return "\n".join("".join(str(v) for v in row) for row in grid)


# Reasoning formats of the worked examples (all end in the same <answer> block):
#   steps    "Step k: r1c2 = 5" plus the whole grid after every step
#   compact  "r1c2=5", one cell per line
#   seen     "r1c2 sees 1,3,4,... -> 5": the digits already in its row, column and box
SHOT_FORMATS = ("steps", "compact", "seen")


def _worked_example(case, style="steps"):
    """Assistant reply: one fill per step, written as `style`, then the answer."""
    if style not in SHOT_FORMATS:
        raise ValueError(f"style must be one of {', '.join(SHOT_FORMATS)}.")
    grid = [row[:] for row in case["puzzle"]]
    digits = range(1, case["size"] + 1)
    parts = []
    steps = solution_steps(grid, case["expected_solution"], case)
    for k, (r, c, value) in enumerate(steps, 1):
        cell = f"r{r + 1}c{c + 1}"
        if style == "seen":
            candidates = cell_candidates(grid, r, c, case)
            seen = ",".join(str(d) for d in digits if d not in candidates)
        grid[r][c] = value
        if style == "steps":
            parts.append(f"Step {k}: {cell} = {value}\n{_grid_lines(grid)}")
        elif style == "compact":
            parts.append(f"{cell}={value}")
        else:
            parts.append(f"{cell} sees {seen} -> {value}")
    parts.append(f"<answer>{json.dumps(grid)}</answer>")
    return ("\n\n" if style == "steps" else "\n").join(parts)


def make_shots(base, test_puzzle, n_shots, seed=0, style="steps"):
    """`n_shots` worked examples on random puzzles different from the test one."""
    if style not in SHOT_FORMATS:
        raise ValueError(f"style must be one of {', '.join(SHOT_FORMATS)}.")
    shots = []
    attempt = 0
    while len(shots) < n_shots:
        attempt += 1
        shot_seed = seed * 1000 + attempt
        solution = random_solution(base, shot_seed)
        data = with_empty_cells(
            {**base, "expected_solution": solution}, SHOT_EMPTY_CELLS, shot_seed
        )
        if data["puzzle"] == test_puzzle:
            continue
        shots.append((build_prompt(data), _worked_example(data, style)))
    return tuple(shots)


def make_case(kind, difficulty, empty_cells=None, seed=0, shots=0, shot_seed=0,
              shot_format="steps"):
    """Registered case, optionally with `empty_cells` blanks and `shots` worked examples.

    With `empty_cells=None` the fixed puzzle is used. Shots add examples;
    the test prompt and final-answer contract are identical with or without them.
    """
    base = prompts[kind][difficulty].data
    data = base if empty_cells is None else with_empty_cells(base, empty_cells, seed)
    return PromptCase(
        prompt=build_prompt(data),
        data=data,
        shots=make_shots(base, data["puzzle"], shots, shot_seed, shot_format),
    )
