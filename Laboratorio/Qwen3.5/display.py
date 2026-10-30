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
    print("\nSudoku:")
    print(format_grid(case.data["puzzle"]))
    print("\nExpected solution:")
    print(format_grid(case.data["expected_solution"]))
    print("\nModel's solution:")
    print(format_grid(report["parsed"].grid))
    print("\nModel output:")
    print(result.text)
    print(f"\nResult: {'PASSED' if report['passed'] else 'FAIL'}")
