"""Named puzzle cases. Add entries as new cases are curated."""

from dataclasses import dataclass
from pathlib import Path

from sudoku import build_prompt, load_sudoku, with_empty_cells


@dataclass(frozen=True)
class PromptCase:
    prompt: str
    data: dict


def _case(filename):
    data = load_sudoku(Path(__file__).with_name(filename))
    return PromptCase(prompt=build_prompt(data), data=data)


# The current puzzle is labeled hard for selection; difficulty has not yet
# been calibrated against the target model.
prompts = {"sudoku9x9": {"hard": _case("sudoku.json")}}


def make_case(kind, difficulty, empty_cells, seed=0):
    """Variant of a registered case with exactly `empty_cells` blank cells."""
    data = with_empty_cells(prompts[kind][difficulty].data, empty_cells, seed)
    return PromptCase(prompt=build_prompt(data), data=data)
