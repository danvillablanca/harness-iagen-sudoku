"""Readable experiment results for the terminal and Colab."""


def format_grid(grid):
    if grid is None:
        return "No parseable solution returned."
    return "\n".join(" ".join(str(value) for value in row) for row in grid)


def print_result(case, report, result, max_output_tokens, load_seconds, generation_seconds):
    print(f"Model loading time: {load_seconds:.2f} seconds")
    print(f"Solving time: {generation_seconds:.2f} seconds")
    print(f"Generated tokens: {result.generated_tokens} / {max_output_tokens}")
    print(f"Finish reason: {result.finish_reason or 'unavailable'}")
    print(f"Output token limit reached: {'YES' if result.generated_tokens >= max_output_tokens else 'NO'}")
    print(f"Shots: {len(case.shots)}")
    print("\nSudoku:")
    print(format_grid(case.data["puzzle"]))
    print("\nReference solution (any valid solution passes):")
    print(format_grid(case.data["expected_solution"]))
    print("\nModel's solution:")
    print(format_grid(report["parsed"].grid))
    print("\nModel input:")
    for user_text, assistant_text in case.shots:
        print(f"[user]\n{user_text}\n\n[assistant]\n{assistant_text}\n")
    print(f"[user]\n{case.prompt}")
    print("\nModel output:")
    print(result.text)
    print("\nVerification checks:")
    for name, passed in report["checks"].items():
        status = "OK" if passed is True else "FAIL" if passed is False else "NOT EVALUATED"
        print(f"  {name}: {status}")
    correctness = (
        "NOT EVALUATED" if report["parsed"].grid is None
        else "CORRECT" if report["solved"] else "INCORRECT"
    )
    print(f"Sudoku correctness: {correctness}")
    for issue in report["parsed"].issues:
        print(f"Parser issue: {issue}")
    score = report["score"]
    parts = {
        "format": int(score["format"]),
        "clues_frac": score["clues_frac"],
        "fill": score["fill"],
        "solved": int(score["solved"]),
    }
    shown = ", ".join(
        f"{name}={'n/a' if value is None else f'{value:.3f}' if isinstance(value, float) else value}"
        for name, value in parts.items()
    )
    print(f"Reward: {report['reward']:.3f} ({shown})")
    print(f"\nResult: {'PASSED' if report['passed'] else 'FAIL'}")
