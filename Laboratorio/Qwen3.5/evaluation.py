"""Parse and evaluate a model response against a puzzle case."""

from response_parser import parse_answer
from scoring import DEFAULT_WEIGHTS, score_response
from sudoku import evaluate_grid


def evaluate(response, case):
    return evaluate_response(response, case.data)


def evaluate_response(response, data, weights=DEFAULT_WEIGHTS):
    """Evaluate text using only the puzzle and its public rules.

    `passed` is the binary full-acceptance flag. `score` holds the partial
    credit components and `reward` is its final value.
    """
    parsed = parse_answer(response, data["size"])
    report = evaluate_grid(parsed.grid, data)
    checks = {
        "Required answer format": parsed.format_ok,
        "Parseable NxN integer array": parsed.grid is not None,
        **report["checks"],
    }
    solved = all(value is True for value in report["checks"].values())
    passed = all(value is True for value in checks.values())
    score = score_response(parsed, report, data, weights)
    return {**report, "checks": checks, "parsed": parsed,
            "solved": solved, "passed": passed,
            "score": score, "reward": score["reward"]}
