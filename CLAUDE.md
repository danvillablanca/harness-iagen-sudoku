# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Sudoku evaluation harness for the course IIC3694-1 (Inteligencia Artificial Generativa): it loads a local Qwen model (default `Qwen/Qwen3.5-2B`), prompts it to solve a Sudoku, and verifies the answer. `informe_IIC3694/` holds the LaTeX report; `Documentos Curso/` holds reference PDFs; `Colab_Sudoku.ipynb` is a Colab version of the harness. Code comments, docs and commit messages are a mix of Spanish (README, commits) and English (code).

## Commands

Python 3.13 (64-bit), virtualenv recommended (`.venv/` is gitignored). Run everything from the repo root:

```sh
python -m pip install -r requirements.txt
python Laboratorio/Qwen3.5/main.py     # exit code 0 iff the answer passed all checks
# CLI flags: --thinking/--no-thinking, --max-output-tokens, --max-prompt-tokens,
#            --empty-cells N, --seed, --shots N, --shot-seed
python Laboratorio/Qwen3.5/dataset.py --out data/sudoku   # train/eval/test JSONL
python Laboratorio/Qwen3.5/baseline.py run --tasks data/sudoku/eval.jsonl --per-level 5 --n 8
python Laboratorio/Qwen3.5/baseline.py summary results/*.jsonl
python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"
cd informe_IIC3694 && latexmk -pdf -interaction=nonstopmode -halt-on-error informe.tex
```

Tests use the standard-library `unittest` (`test_*.py` next to the modules); there is no configured linter. The modules in `Laboratorio/Qwen3.5/` use flat imports (`from config import ...`), so they must be run with that directory as the script directory (as `main.py` does), not as a package.

## Architecture (`Laboratorio/Qwen3.5/`)

Pipeline wired in `main.py` (argparse flags above): `ModelConfig` → `make_case(...)` → `load_model` → `model.generate_with_stats(case.prompt, shots=case.shots)` → `evaluate(text, case)` → `print_result`. `--max-prompt-tokens` defaults to 2048, or 16384 when `--shots > 0`.

- `config.py` — frozen `ModelConfig` (model id, `backend` auto/mlx/cuda/mps/cpu, `thinking`, prompt/output token limits) with validation in `__post_init__`.
- `models.py` — public `load_model` entry point; currently delegates to `qwen.py`. `qwen.py` owns backend selection, resource checks (RAM/VRAM per the README table), and generation; `generate` returns text, `generate_with_stats` returns a `GenerationResult` (tokens, finish reason). Both accept per-call `max_tokens` / `enable_thinking` overrides and `shots`, a sequence of (user, assistant) turns placed before the prompt in the chat template. `generate_samples(prompt, n, temperature=..., top_p=..., seed=...)` returns `n` sampled `GenerationResult`s (one batch with `num_return_sequences` on PyTorch, sequential on MLX); `_torch_result` trims batch padding after the first EOS.
- `baseline.py` — `run` samples `n` completions per task from a `dataset.py` JSONL, scores them with `reward`, and appends one line per sample to `results/<model>-<thinking|nothinking>.jsonl`; reruns resume and refuse mismatched settings (`RUN_KEYS`). `summary` computes unbiased `pass_at_k` per (model, thinking, size, level), plus truncation rate and mean tokens, as a Markdown table.
- `prompts.py` — registry `prompts[kind][difficulty]` of `PromptCase(prompt, data)`, built from JSON files via `sudoku.load_sudoku` + `build_prompt`. Add new puzzles here. The `"hard"` label is not yet calibrated experimentally. `make_case(kind, difficulty, empty_cells, seed, shots, shot_seed)` returns a `PromptCase` (with a `shots` tuple): `empty_cells` blanks random cells of the solution via `sudoku.with_empty_cells` (`None` keeps the fixed puzzle); `make_shots` builds few-shot examples on random puzzles (`SHOT_EMPTY_CELLS = 8`, different from the test puzzle) whose replies fill one cell per step (`Step k: rXcY = v` plus the grid) and end in `<answer>`. Shots never change the test prompt or its response contract.
- `sudoku.json` — a case: `size`, `box_rows`/`box_columns`, `puzzle` (0 = empty), `expected_solution`, and one `prompt_template` (formatted with `{size}` etc. by `build_prompt`). Explanations and intermediate grids are optional; only the final answer is scored.
- `response_parser.py` — `parse_answer` extracts an NxN grid from one final `<answer>` block, allowing explanations before it regardless of shots. Extra text after it, ambiguous answer blocks, malformed thinking delimiters and answer blocks inside thinking are rejected. Bare JSON or a standalone Markdown fence can be recovered for diagnostics only.
- `dataset.py` — task splits for training and evaluation, adapted from Reasoning Gym's Sudoku generator: 4x4/6x6/9x9 boards, `LEVELS` of blank-cell ranges per size (uncalibrated), cells removed one by one while the solution stays unique (`--no-unique` to skip). Each record is seeded by `(seed, split, size, level, index, attempt)`; splits are generated test → eval → train and never share a puzzle. Writes `train/eval/test.jsonl`; `case_from_record` adapts a record to `PromptCase`.
- `sudoku.py` — case loading/validation, puzzle generation helpers (`random_solution`, `with_empty_cells`, `solution_steps`, `is_deducible`, `count_solutions`) and `evaluate_grid`; `with_empty_cells` resamples blanks until the puzzle is solvable by only filling cells with a single candidate. `evaluate_grid` produces the named checks in `CHECK_NAMES` (rows, columns, boxes, clues preserved; any valid completion passes, uniqueness is not required). `evaluation.py` merges the parser's format checks with these; `passed` requires every check to be `True`.
- `display.py` — `print_result` prints timings, token stats, shot count, puzzle/expected/parsed grids, the full model input (shots included) and output, and PASSED/FAIL.
- `reward.py` — model-independent `reward(prompt, completion) -> float`, currently binary. `sudoku.task_from_prompt` reads the public task from the `build_prompt` text; invalid prompts raise `ValueError`, while invalid completions get zero. Reference solutions are not used. `evaluation.evaluate_response` separates mathematical correctness (`solved`), full acceptance (`passed`), and `reward`. Partial credit and difficulty weighting are pending, not silently approximated.
- `timing.py` — `timed(fn, *args)` returns `(result, seconds)`.
