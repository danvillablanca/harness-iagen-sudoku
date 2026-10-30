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
cd informe_IIC3694 && latexmk -pdf -interaction=nonstopmode -halt-on-error informe.tex
```

There is no test suite or linter. The modules in `Laboratorio/Qwen3.5/` use flat imports (`from config import ...`), so they must be run with that directory as the script directory (as `main.py` does), not as a package.

## Architecture (`Laboratorio/Qwen3.5/`)

Pipeline wired in `main.py` (argparse flags above): `ModelConfig` → `make_case(...)` → `load_model` → `model.generate_with_stats(case.prompt, shots=case.shots)` → `evaluate(text, case)` → `print_result`. `--max-prompt-tokens` defaults to 2048, or 16384 when `--shots > 0`.

- `config.py` — frozen `ModelConfig` (model id, `backend` auto/mlx/cuda/mps/cpu, `thinking`, prompt/output token limits) with validation in `__post_init__`.
- `models.py` — public `load_model` entry point; currently delegates to `qwen.py`. `qwen.py` owns backend selection, resource checks (RAM/VRAM per the README table), and generation; `generate` returns text, `generate_with_stats` returns a `GenerationResult` (tokens, finish reason). Both accept per-call `max_tokens` / `enable_thinking` overrides and `shots`, a sequence of (user, assistant) turns placed before the prompt in the chat template.
- `prompts.py` — registry `prompts[kind][difficulty]` of `PromptCase(prompt, data)`, built from JSON files via `sudoku.load_sudoku` + `build_prompt`. Add new puzzles here. The `"hard"` label is not yet calibrated experimentally. `make_case(kind, difficulty, empty_cells, seed, shots, shot_seed)` returns a `PromptCase` (with a `shots` tuple): `empty_cells` blanks random cells of the solution via `sudoku.with_empty_cells` (`None` keeps the fixed puzzle); `make_shots` builds few-shot examples on random puzzles (`SHOT_EMPTY_CELLS = 8`, different from the test puzzle) whose replies fill one cell per step (`Step k: rXcY = v` plus the grid) and end in `<answer>`. With shots, the prompt uses `steps_prompt_template`.
- `sudoku.json` — a case: `size`, `box_rows`/`box_columns`, `puzzle` (0 = empty), `expected_solution`, `prompt_template` (formatted with `{size}` etc. by `build_prompt`) and `steps_prompt_template` (step-by-step variant used with shots).
- `response_parser.py` — `parse_answer` extracts an NxN grid from the model response and records whether the required answer format was followed; with `allow_steps=True` (set by `evaluate` when the case has shots) work before the final `<answer>` block is allowed.
- `sudoku.py` — case loading/validation, puzzle generation helpers (`random_solution`, `with_empty_cells`, `solution_steps`, `count_solutions`) and `evaluate_grid`, which produces the named checks in `CHECK_NAMES` (rows, columns, boxes, clues preserved, matches reference, ...). `evaluation.py` merges the parser's format checks with these; `passed` requires every check to be `True`.
- `display.py` — `print_result` prints timings, token stats, shot count, puzzle/expected/parsed grids, the full model input (shots included) and output, and PASSED/FAIL.
- `timing.py` — `timed(fn, *args)` returns `(result, seconds)`.
