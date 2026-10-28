from config import ModelConfig
from evaluation import evaluate
from models import load_model
from prompts import prompts
from timing import timed


def main() -> int:
    config = ModelConfig(thinking=False, max_output_tokens=2048)
    case = prompts["sudoku9x9"]["hard"]

    model, load_seconds = timed(load_model, config)
    result, generation_seconds = timed(model.generate_with_stats, case.prompt)
    report, validation_seconds = timed(evaluate, result.text, case)

    print(f"Model loaded in {load_seconds:.2f} seconds ({model.backend}).")
    print(f"Generation time: {generation_seconds:.2f} seconds.")
    print(f"Validation time: {validation_seconds:.4f} seconds.")
    print(f"Tokens: {result.prompt_tokens} prompt, {result.generated_tokens} generated.")
    print(f"Finish reason: {result.finish_reason or 'unavailable'}.")
    print("\n--- Model response ---")
    print(result.text)
    print("\n--- Validation ---")
    print(f"Parser source: {report['parsed'].source}")
    for issue in report["parsed"].issues:
        print(f"Format detail: {issue}")
    for name, passed in report["checks"].items():
        status = "NOT CHECKED" if passed is None else "PASS" if passed else "FAIL"
        print(f"{status}: {name}")
    print(f"FINAL RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
