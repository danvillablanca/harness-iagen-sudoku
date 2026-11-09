"""Partial-credit score of one parsed response. See REWARD.md for the rationale.

    reward = format * (w_fill * fill * clues_frac + w_solve * solved)

`passed` stays binary and separate; this module only adds the partial score.
"""

import math
from dataclasses import dataclass

from sudoku import unique_solution


@dataclass(frozen=True)
class RewardWeights:
    fill: float = 0.5
    solve: float = 0.5

    def __post_init__(self):
        for name in ("fill", "solve"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"The {name} weight must be a finite number >= 0.")
        if not math.isclose(self.fill + self.solve, 1.0, abs_tol=1e-9):
            raise ValueError("The fill and solve weights must sum to 1.")


DEFAULT_WEIGHTS = RewardWeights()


def score_response(parsed, report, data, weights=DEFAULT_WEIGHTS) -> dict:
    """Score a parsed response against the public puzzle in `data`.

    `report` is `evaluate_grid`'s output for `parsed.grid`. Returns:
      format      the required output format was met
      clues_frac  original clues kept / total clues (None if not computed)
      fill        correct originally-empty cells / empty cells, against the
                  unique solution (None if the puzzle is not provably unique,
                  or if the format failed)
      solved      the grid satisfies every rule and keeps every clue
      exact       the puzzle has a proven unique solution (None if not computed)
      reward      float in [0, 1]

    Without a proven unique solution there is no partial credit:
    reward = format * solved.
    """
    solved = all(value is True for value in report["checks"].values())
    if not parsed.format_ok or parsed.grid is None or report["clues_preserved"] is None:
        return {"format": False, "clues_frac": None, "fill": None,
                "solved": solved, "exact": None, "reward": 0.0}

    total_clues = report["total_clues"]
    clues_frac = report["clues_preserved"] / total_clues if total_clues else 1.0

    puzzle = data["puzzle"]
    solution = unique_solution(puzzle, data)
    exact = solution is not None
    fill = None
    if exact:
        empty = [
            (r, c)
            for r, row in enumerate(puzzle)
            for c, value in enumerate(row)
            if value == 0
        ]
        fill = (
            sum(parsed.grid[r][c] == solution[r][c] for r, c in empty) / len(empty)
            if empty else 1.0
        )
        reward = weights.fill * fill * clues_frac + weights.solve * solved
    else:
        reward = float(solved)

    if solved:
        # The weights sum to 1. Snap to avoid floating-point drift, so that
        # reward == 1.0 exactly when the response is solved (baseline.py's
        # pass@k relies on that).
        reward = 1.0
    return {"format": True, "clues_frac": clues_frac, "fill": fill,
            "solved": solved, "exact": exact, "reward": min(1.0, reward)}
