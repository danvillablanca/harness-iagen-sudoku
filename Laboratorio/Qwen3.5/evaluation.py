"""Parse and evaluate a model response against a puzzle case."""

from response_parser import parse_answer
from sudoku import evaluate_grid


def evaluate(response, case):
    return evaluate_response(response, case.data)


def evaluate_response(response, data):
    """Evaluate text using only the puzzle and its public rules."""
    parsed = parse_answer(response, data["size"])
    report = evaluate_grid(parsed.grid, data)
    checks = {
        "Required answer format": parsed.format_ok,
        "Parseable NxN integer array": parsed.grid is not None,
        **report["checks"],
    }
    solved = all(value is True for value in report["checks"].values())
    passed = all(value is True for value in checks.values())
    return {**report, "checks": checks, "parsed": parsed,
            "solved": solved, "passed": passed, "reward": float(passed)}
