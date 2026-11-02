"""Course-compatible binary reward, independent of model loading."""

from evaluation import evaluate_response
from sudoku import task_from_prompt


def reward(prompt: str, completion: str) -> float:
    """Return 1.0 for one valid final solution in the required format, else 0.0.

    The prompt must follow build_prompt's layout. Malformed prompts raise
    ValueError; malformed completions receive zero. Any mathematically valid
    completion preserving the original clues is accepted, without consulting
    an expected solution. Partial credit and difficulty weighting are not used.
    """
    return evaluate_response(completion, task_from_prompt(prompt))["reward"]
