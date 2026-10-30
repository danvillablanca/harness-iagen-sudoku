import argparse

from config import ModelConfig
from display import print_result
from evaluation import evaluate
from models import load_model
from prompts import make_case, prompts
from timing import timed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Sudoku harness.")
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable Qwen's reasoning mode (default: enabled)",
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--empty-cells",
        type=int,
        default=None,
        help="number of blank cells (default: the fixed puzzle in sudoku.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for choosing which cells to blank with --empty-cells",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ModelConfig(thinking=args.thinking, max_output_tokens=args.max_output_tokens)
    if args.empty_cells is None:
        case = prompts["sudoku9x9"]["hard"]
    else:
        case = make_case("sudoku9x9", "hard", args.empty_cells, args.seed)

    model, load_seconds = timed(load_model, config)
    result, generation_seconds = timed(model.generate_with_stats, case.prompt)
    report = evaluate(result.text, case)
    print_result(case, report, result, config.max_output_tokens, load_seconds, generation_seconds)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
