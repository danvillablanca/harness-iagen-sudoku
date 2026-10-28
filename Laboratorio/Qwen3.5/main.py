# main.py

import time
from pathlib import Path

from qwen import load_model
from response_parser import parse_answer
from sudoku import build_prompt, evaluate_grid, load_sudoku


ENABLE_THINKING = False  # Change to True to enable thinking mode.
MAX_TOKENS = 2048

# Resolve relative to main.py, regardless of the terminal's working directory.
CASE_PATH = Path(__file__).with_name("sudoku.json")


def status(value):
    if value is None:
        return "NOT CHECKED"
    return "PASS" if value else "FAIL"


def main() -> int:
    total_start = time.perf_counter()

    case = load_sudoku(CASE_PATH)
    prompt = build_prompt(case)

    started = time.perf_counter()
    model = load_model()
    load_seconds = time.perf_counter() - started

    print(f"Model loaded in {load_seconds:.2f} seconds.")
    print(f"Backend: {model.backend}")
    print(f"Thinking mode: {'ON' if ENABLE_THINKING else 'OFF'}")
    print("Generating Sudoku answer...")

    started = time.perf_counter()
    result = model.generate_with_stats(
        prompt,
        max_tokens=MAX_TOKENS,
        enable_thinking=ENABLE_THINKING,
    )
    generation_seconds = time.perf_counter() - started

    print(f"Answer generated in {generation_seconds:.2f} seconds.")
    print("\n--- Model response ---")
    print(result.text)

    print("\n--- Token statistics ---")
    print(f"Prompt tokens: {result.prompt_tokens}")
    print(f"Generated tokens: {result.generated_tokens}")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Output token budget: {MAX_TOKENS}")

    source = "runtime" if result.backend == "mlx" else "inferred"
    print(
        f"Finish reason ({source}): "
        f"{result.finish_reason or 'unavailable'}"
    )
    print(
        "Output budget reached: "
        f"{result.generated_tokens >= MAX_TOKENS}"
    )

    if generation_seconds > 0:
        print(
            "End-to-end output tokens/second: "
            f"{result.generated_tokens / generation_seconds:.2f}"
        )

    started = time.perf_counter()

    parsed = parse_answer(result.text, case["size"])
    report = evaluate_grid(parsed.grid, case)

    checks = {
        "Required answer format": parsed.format_ok,
        "Parseable NxN integer array": parsed.grid is not None,
        **report["checks"],
    }

    validation_seconds = time.perf_counter() - started

    print("\n--- Validation ---")
    print(f"Parser source: {parsed.source}")

    for issue in parsed.issues:
        print(f"Format detail: {issue}")

    for label, passed in checks.items():
        print(f"{status(passed)}: {label}")

    if parsed.grid is not None:
        print(
            "Copied the original puzzle: "
            f"{report['copied_input']}"
        )
        print(
            "Empty cells remaining: "
            f"{report['empty_cells_remaining']}"
        )
        print(
            "Originally empty cells filled: "
            f"{report['filled_empty_cells']}/"
            f"{report['initial_empty_cells']}"
        )
        print(
            "Original clues preserved: "
            f"{report['clues_preserved']}/"
            f"{report['total_clues']}"
        )

    passed = all(value is True for value in checks.values())

    print(f"Validation time: {validation_seconds:.4f} seconds.")
    print(
        "Total execution time: "
        f"{time.perf_counter() - total_start:.2f} seconds."
    )
    print(f"FINAL RESULT: {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())