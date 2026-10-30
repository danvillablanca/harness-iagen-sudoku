"""Parse and evaluate a model response against a puzzle case."""

from response_parser import parse_answer
from sudoku import evaluate_grid


def evaluate(response, case):
    parsed = parse_answer(response, case.data["size"], allow_steps=bool(case.shots))
    report = evaluate_grid(parsed.grid, case.data)
    checks = {
        "Required answer format": parsed.format_ok,
        "Parseable NxN integer array": parsed.grid is not None,
        **report["checks"],
    }
    return {**report, "checks": checks, "parsed": parsed,
            "passed": all(value is True for value in checks.values())}
