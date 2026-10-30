"""Named puzzle cases. Add entries as new cases are curated."""

from dataclasses import dataclass
from pathlib import Path

import json

from sudoku import (
    build_prompt,
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


def _worked_example(case):
    """Assistant reply: one fill per step with the grid after it, then the answer."""
    grid = [row[:] for row in case["puzzle"]]
    parts = []
    steps = solution_steps(grid, case["expected_solution"], case)
    for k, (r, c, value) in enumerate(steps, 1):
        grid[r][c] = value
        parts.append(f"Step {k}: r{r + 1}c{c + 1} = {value}\n{_grid_lines(grid)}")
    parts.append(f"<answer>{json.dumps(grid)}</answer>")
    return "\n\n".join(parts)


def make_shots(base, test_puzzle, n_shots, seed=0):
    """`n_shots` worked examples on random puzzles different from the test one."""
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
        shots.append((build_prompt(data, with_steps=True), _worked_example(data)))
    return tuple(shots)


def make_case(kind, difficulty, empty_cells=None, seed=0, shots=0, shot_seed=0):
    """Registered case, optionally with `empty_cells` blanks and `shots` worked examples.

    With `empty_cells=None` the fixed puzzle is used. With shots, the prompt
    asks for step-by-step filling so it matches the examples.
    """
    base = prompts[kind][difficulty].data
    data = base if empty_cells is None else with_empty_cells(base, empty_cells, seed)
    return PromptCase(
        prompt=build_prompt(data, with_steps=shots > 0),
        data=data,
        shots=make_shots(base, data["puzzle"], shots, shot_seed),
    )
