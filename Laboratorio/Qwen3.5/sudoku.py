# sudoku.py

import json
import random
import re
from functools import lru_cache
from pathlib import Path
from string import Formatter


CHECK_NAMES = (
    "All cells filled",
    "Allowed values",
    "Rows",
    "Columns",
    "Boxes",
    "Original clues",
)


def _shape_ok(grid, size):
    return (
        isinstance(grid, list)
        and len(grid) == size
        and all(
            isinstance(row, list) and len(row) == size
            for row in grid
        )
    )


def load_sudoku(path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        case = json.load(handle)

    if not isinstance(case, dict):
        raise ValueError("The Sudoku configuration must be a JSON object.")

    required = {
        "name",
        "size",
        "box_rows",
        "box_columns",
        "puzzle",
        "expected_solution",
        "prompt_template",
    }

    if missing := required - case.keys():
        raise ValueError(
            f"Missing configuration fields: {', '.join(sorted(missing))}."
        )

    for key in ("name", "prompt_template"):
        if not isinstance(case[key], str) or not case[key].strip():
            raise ValueError(f"{key} must be a nonempty string.")

    for key in ("size", "box_rows", "box_columns"):
        if type(case[key]) is not int or case[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    size = case["size"]

    if case["box_rows"] * case["box_columns"] != size:
        raise ValueError("box_rows * box_columns must equal size.")

    for key, minimum in (("puzzle", 0), ("expected_solution", 1)):
        grid = case[key]

        if not _shape_ok(grid, size):
            raise ValueError(f"{key} must be a {size}x{size} array.")

        if any(
            type(value) is not int or not minimum <= value <= size
            for row in grid
            for value in row
        ):
            raise ValueError(
                f"{key} must contain integers from {minimum} to {size}."
            )

    reference_checks = evaluate_grid(
        case["expected_solution"], case
    )["checks"]

    failed = [
        name
        for name, passed in reference_checks.items()
        if not passed
    ]

    if failed:
        raise ValueError(
            "Invalid reference solution: " + ", ".join(failed)
        )

    # Validate the templates before loading the model.
    build_prompt(case)

    return case


def build_prompt(case: dict) -> str:
    puzzle_text = (
        "[\n"
        + ",\n".join(
            "  " + json.dumps(row)
            for row in case["puzzle"]
        )
        + "\n]"
    )

    # Only these values are available to the prompt template.
    # expected_solution is deliberately excluded.
    values = {
        "size": case["size"],
        "box_rows": case["box_rows"],
        "box_columns": case["box_columns"],
        "puzzle": puzzle_text,
    }

    template = case["prompt_template"]

    for _, field, spec, conversion in Formatter().parse(template):
        if field is not None and (
            field not in values or spec or conversion
        ):
            raise ValueError(
                "Prompt placeholders must be size, box_rows, "
                "box_columns, or puzzle."
            )

    return template.format(**values)


def task_from_prompt(prompt: str) -> dict:
    """Recover public task data from the prompt emitted by build_prompt.

    No reference solution is needed by the reward function. Invalid prompts
    are harness configuration errors, rather than failed model responses.
    """
    if not isinstance(prompt, str):
        raise ValueError("The prompt must be text.")
    dimensions = re.search(r"\b(\d+)x(\d+) Sudoku\b", prompt)
    boxes = re.search(r"\bEach (\d+)x(\d+) box\b", prompt)
    marker = "\nPuzzle:\n"
    if dimensions is None or boxes is None or prompt.count(marker) != 1:
        raise ValueError("Expected a Sudoku prompt with dimensions, boxes, and one puzzle.")
    size, columns = map(int, dimensions.groups())
    box_rows, box_columns = map(int, boxes.groups())
    if (
        size <= 0 or columns != size or box_rows <= 0 or box_columns <= 0
        or box_rows * box_columns != size
    ):
        raise ValueError("Invalid Sudoku dimensions in the prompt.")
    payload = prompt.split(marker, 1)[1].lstrip()
    try:
        puzzle, _ = json.JSONDecoder().raw_decode(payload)
    except ValueError as error:
        raise ValueError("The prompt puzzle must be a JSON array.") from error
    if not _shape_ok(puzzle, size) or any(
        type(value) is not int or not 0 <= value <= size
        for row in puzzle for value in row
    ):
        raise ValueError("Invalid puzzle shape or values in the prompt.")
    groups = [*puzzle, *zip(*puzzle)]
    groups.extend(
        [puzzle[r][c]
         for r in range(r0, r0 + box_rows)
         for c in range(c0, c0 + box_columns)]
        for r0 in range(0, size, box_rows)
        for c0 in range(0, size, box_columns)
    )
    for group in groups:
        values = [value for value in group if value]
        if len(values) != len(set(values)):
            raise ValueError("The prompt contains conflicting Sudoku clues.")
    return dict(size=size, box_rows=box_rows, box_columns=box_columns, puzzle=puzzle)


def _candidates(grid, r, c, case):
    size = case["size"]
    box_rows = case["box_rows"]
    box_columns = case["box_columns"]
    used = set(grid[r]) | {grid[i][c] for i in range(size)}
    r0 = r - r % box_rows
    c0 = c - c % box_columns
    used |= {
        grid[i][j]
        for i in range(r0, r0 + box_rows)
        for j in range(c0, c0 + box_columns)
    }
    return [v for v in range(1, size + 1) if v not in used]


def _most_constrained_cell(grid, case):
    """Empty cell with the fewest candidates as (r, c, options), or None."""
    size = case["size"]
    best = None
    for r in range(size):
        for c in range(size):
            if grid[r][c] == 0:
                options = _candidates(grid, r, c, case)
                if best is None or len(options) < len(best[2]):
                    best = (r, c, options)
    return best


def random_solution(case: dict, seed: int | None = None) -> list[list[int]]:
    """Random completed grid obeying the rules of `case`, by backtracking."""
    size = case["size"]
    rng = random.Random(seed)
    grid = [[0] * size for _ in range(size)]

    def fill():
        best = _most_constrained_cell(grid, case)
        if best is None:
            return True
        r, c, options = best
        rng.shuffle(options)
        for value in options:
            grid[r][c] = value
            if fill():
                return True
            grid[r][c] = 0
        return False

    fill()
    return grid


def solution_steps(
    puzzle: list[list[int]], solution: list[list[int]], case: dict
) -> list[tuple[int, int, int]]:
    """Order in which to fill the blanks: most constrained cell first."""
    grid = [row[:] for row in puzzle]
    steps = []
    while (best := _most_constrained_cell(grid, case)) is not None:
        r, c, _ = best
        grid[r][c] = solution[r][c]
        steps.append((r, c, solution[r][c]))
    return steps


def is_deducible(puzzle: list[list[int]], case: dict) -> bool:
    """True if repeatedly filling an empty cell that has exactly one
    candidate completes the grid (no guessing needed)."""
    size = case["size"]
    grid = [row[:] for row in puzzle]
    while True:
        empty = [
            (r, c) for r in range(size) for c in range(size) if grid[r][c] == 0
        ]
        if not empty:
            return True
        for r, c in empty:
            options = _candidates(grid, r, c, case)
            if len(options) == 1:
                grid[r][c] = options[0]
                break
        else:
            return False


def find_solutions(
    puzzle: list[list[int]], case: dict, limit: int = 2
) -> list[list[list[int]]]:
    """Valid completions of `puzzle`, stopping after `limit` of them."""
    grid = [row[:] for row in puzzle]
    found = []

    def search():
        best = _most_constrained_cell(grid, case)
        if best is None:
            found.append([row[:] for row in grid])
            return
        r, c, options = best
        for value in options:
            grid[r][c] = value
            search()
            grid[r][c] = 0
            if len(found) >= limit:
                return

    search()
    return found


def count_solutions(puzzle: list[list[int]], case: dict, limit: int = 2) -> int:
    """Number of valid completions of `puzzle`, counting at most `limit`.

    With the default limit, 1 means the solution is unique and 2 means
    there are at least two."""
    return len(find_solutions(puzzle, case, limit))


def unique_solution(
    puzzle: list[list[int]], case: dict
) -> tuple[tuple[int, ...], ...] | None:
    """The only completion of `puzzle`, or None if it has none or several.

    Uses only the public rules, never `expected_solution`. The search is
    exhaustive, so a returned grid is a proof of uniqueness. Results are cached
    per puzzle: every sample of a task shares the same puzzle."""
    return _unique_solution(
        case["size"],
        case["box_rows"],
        case["box_columns"],
        tuple(tuple(row) for row in puzzle),
    )


@lru_cache(maxsize=4096)
def _unique_solution(size, box_rows, box_columns, puzzle):
    case = {"size": size, "box_rows": box_rows, "box_columns": box_columns}
    found = find_solutions([list(row) for row in puzzle], case)
    if len(found) != 1:
        return None
    return tuple(tuple(row) for row in found[0])


MAX_BLANKING_ATTEMPTS = 1000


def with_empty_cells(
    case: dict,
    empty_cells: int,
    seed: int | None = None,
) -> dict:
    """Return a copy of `case` whose puzzle is the reference solution with
    `empty_cells` random cells blanked, resampled until the puzzle can be
    solved by only filling cells with a single candidate. Such a puzzle always
    has a unique solution, because every step is forced. Any grid passing the
    rule checks is accepted regardless."""
    size = case["size"]

    if not 0 <= empty_cells <= size * size:
        raise ValueError(
            f"empty_cells must be between 0 and {size * size}."
        )

    rng = random.Random(seed)
    positions = [(r, c) for r in range(size) for c in range(size)]
    for _ in range(MAX_BLANKING_ATTEMPTS):
        puzzle = [row[:] for row in case["expected_solution"]]
        for r, c in rng.sample(positions, empty_cells):
            puzzle[r][c] = 0
        if is_deducible(puzzle, case):
            return {**case, "puzzle": puzzle}
    raise ValueError(
        f"Could not blank {empty_cells} cells so that every step has a "
        f"single candidate after {MAX_BLANKING_ATTEMPTS} attempts."
    )


def evaluate_grid(
    grid: list[list[int]] | None,
    case: dict,
) -> dict:
    size = case["size"]
    puzzle = case["puzzle"]

    empty_positions = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if puzzle[r][c] == 0
    ]
    clue_positions = [
        (r, c)
        for r in range(size)
        for c in range(size)
        if puzzle[r][c] != 0
    ]

    report = {
        "checks": dict.fromkeys(CHECK_NAMES),
        "copied_input": None,
        "empty_cells_remaining": None,
        "filled_empty_cells": None,
        "initial_empty_cells": len(empty_positions),
        "clues_preserved": None,
        "total_clues": len(clue_positions),
    }

    # No recovered grid means mathematical checks were not performed.
    if grid is None:
        return report

    if (
        not _shape_ok(grid, size)
        or any(
            type(value) is not int
            for row in grid
            for value in row
        )
    ):
        report.update(
            checks=dict.fromkeys(CHECK_NAMES, False),
            copied_input=False,
        )
        return report

    digits = set(range(1, size + 1))
    box_rows = case["box_rows"]
    box_columns = case["box_columns"]

    preserved = sum(
        grid[r][c] == puzzle[r][c]
        for r, c in clue_positions
    )

    report["checks"] = {
        "All cells filled": all(
            value != 0
            for row in grid
            for value in row
        ),
        "Allowed values": all(
            value in digits
            for row in grid
            for value in row
        ),
        "Rows": all(
            set(row) == digits
            for row in grid
        ),
        "Columns": all(
            {grid[r][c] for r in range(size)} == digits
            for c in range(size)
        ),
        "Boxes": all(
            {
                grid[r][c]
                for r in range(r0, r0 + box_rows)
                for c in range(c0, c0 + box_columns)
            } == digits
            for r0 in range(0, size, box_rows)
            for c0 in range(0, size, box_columns)
        ),
        "Original clues": preserved == len(clue_positions),
    }

    report.update(
        copied_input=grid == puzzle,
        empty_cells_remaining=sum(
            value == 0
            for row in grid
            for value in row
        ),
        filled_empty_cells=sum(
            grid[r][c] in digits
            for r, c in empty_positions
        ),
        clues_preserved=preserved,
    )

    return report
