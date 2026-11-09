"""Course-compatible reward with partial credit, independent of model loading.

The formula and its rationale are in REWARD.md.
"""

from evaluation import evaluate_response
from scoring import DEFAULT_WEIGHTS, RewardWeights
from sudoku import task_from_prompt

__all__ = ["RewardWeights", "make_reward", "reward", "reward_parts"]


def reward_parts(prompt: str, completion: str, weights=DEFAULT_WEIGHTS) -> dict:
    """Components behind the reward: format, clues_frac, fill, solved, exact, reward.

    Malformed prompts raise ValueError; malformed completions score zero.
    """
    return evaluate_response(completion, task_from_prompt(prompt), weights)["score"]


def reward(prompt: str, completion: str) -> float:
    """Return a score in [0, 1]: 1.0 for one valid final solution in the required
    format, partial credit for a well-formed but incomplete or wrong answer.

    The prompt must follow build_prompt's layout. Malformed prompts raise
    ValueError; completions that miss the required format receive zero. Any
    mathematically valid completion preserving the original clues scores 1.0,
    without consulting an expected solution. Partial credit uses the puzzle's
    unique solution, which is proven by search; without a proven unique
    solution the reward is binary.
    """
    return reward_parts(prompt, completion)["reward"]


def make_reward(weights: RewardWeights):
    """Return a reward function with the same signature as `reward`, using `weights`."""
    def custom_reward(prompt: str, completion: str) -> float:
        return reward_parts(prompt, completion, weights)["reward"]
    return custom_reward
