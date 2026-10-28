from config import ModelConfig
from display import print_result
from evaluation import evaluate
from models import load_model
from prompts import prompts
from timing import timed


def main() -> int:
    config = ModelConfig(thinking=True, max_output_tokens=8192)
    case = prompts["sudoku9x9"]["hard"]

    model, load_seconds = timed(load_model, config)
    result, generation_seconds = timed(model.generate_with_stats, case.prompt)
    report = evaluate(result.text, case)
    print_result(case, report, result, config.max_output_tokens, load_seconds, generation_seconds)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
