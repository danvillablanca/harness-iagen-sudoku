"""Tests for the partial-credit reward. Run from the repository root:

    python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"
"""

import json
import unittest

from dataset import board
from evaluation import evaluate_response
from reward import RewardWeights, make_reward, reward, reward_parts
from sudoku import build_prompt, count_solutions, task_from_prompt, unique_solution

# 4x4 puzzle with a single solution: three empty cells, 13 clues.
UNIQUE = [[1, 0, 3, 4], [3, 4, 1, 2], [0, 0, 4, 3], [4, 3, 2, 1]]
SOLUTION = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]

# 4x4 puzzle with two solutions (the 1/2 swap in rows 0 and 2, columns 0 and 1).
AMBIGUOUS = [[0, 0, 3, 4], [3, 4, 1, 2], [0, 0, 4, 3], [4, 3, 2, 1]]
ALTERNATIVE = [[2, 1, 3, 4], [3, 4, 1, 2], [1, 2, 4, 3], [4, 3, 2, 1]]


def answer(grid):
    return f"<answer>{json.dumps(grid)}</answer>"


def prompt_for(puzzle):
    return build_prompt({**board(4), "puzzle": puzzle})


def with_cell(grid, r, c, value):
    copy = [row[:] for row in grid]
    copy[r][c] = value
    return copy


class UniqueSolutionTest(unittest.TestCase):
    def test_puzzles_have_the_expected_number_of_solutions(self):
        self.assertEqual(count_solutions(UNIQUE, board(4)), 1)
        self.assertEqual(count_solutions(AMBIGUOUS, board(4)), 2)

    def test_unique_puzzle_returns_its_solution(self):
        found = unique_solution(UNIQUE, board(4))
        self.assertEqual([list(row) for row in found], SOLUTION)
        self.assertIs(unique_solution(UNIQUE, board(4)), found)  # cached

    def test_ambiguous_puzzle_has_no_unique_solution(self):
        self.assertIsNone(unique_solution(AMBIGUOUS, board(4)))


class RewardWeightsTest(unittest.TestCase):
    def test_defaults_and_valid_weights(self):
        self.assertEqual((RewardWeights().fill, RewardWeights().solve), (0.5, 0.5))
        RewardWeights(0.3, 0.7)
        RewardWeights(1, 0)

    def test_invalid_weights_are_rejected(self):
        for fill, solve in [(-0.1, 1.1), (0.5, 0.6), (float("nan"), 1.0),
                            (True, 0.0), ("0.5", 0.5)]:
            with self.subTest(fill=fill, solve=solve):
                with self.assertRaises(ValueError):
                    RewardWeights(fill, solve)


class RewardTest(unittest.TestCase):
    def test_correct_solution_scores_one(self):
        self.assertEqual(reward(prompt_for(UNIQUE), answer(SOLUTION)), 1.0)

    def test_explanation_before_the_answer_is_allowed(self):
        completion = "Step 1: r1c2 = 2\n" + answer(SOLUTION)
        self.assertEqual(reward(prompt_for(UNIQUE), completion), 1.0)

    def test_unchanged_puzzle_scores_zero(self):
        parts = reward_parts(prompt_for(UNIQUE), answer(UNIQUE))
        self.assertEqual(parts["fill"], 0.0)
        self.assertEqual(parts["reward"], 0.0)

    def test_wrong_cell_gets_partial_credit(self):
        parts = reward_parts(prompt_for(UNIQUE), answer(with_cell(SOLUTION, 2, 1, 2)))
        self.assertTrue(parts["format"])
        self.assertAlmostEqual(parts["fill"], 2 / 3)
        self.assertEqual(parts["clues_frac"], 1.0)
        self.assertFalse(parts["solved"])
        self.assertTrue(parts["exact"])
        self.assertAlmostEqual(parts["reward"], 0.5 * 2 / 3)

    def test_zero_counts_as_an_unfilled_cell_but_keeps_the_format(self):
        parts = reward_parts(prompt_for(UNIQUE), answer(with_cell(SOLUTION, 2, 1, 0)))
        self.assertTrue(parts["format"])
        self.assertAlmostEqual(parts["fill"], 2 / 3)
        self.assertAlmostEqual(parts["reward"], 0.5 * 2 / 3)

    def test_altered_clue_lowers_clues_frac_and_the_reward(self):
        parts = reward_parts(prompt_for(UNIQUE), answer(with_cell(SOLUTION, 0, 0, 2)))
        self.assertEqual(parts["fill"], 1.0)
        self.assertAlmostEqual(parts["clues_frac"], 12 / 13)
        self.assertFalse(parts["solved"])
        self.assertAlmostEqual(parts["reward"], 0.5 * 12 / 13)

    def test_broken_format_scores_zero(self):
        for name, completion in {
            "no tags": json.dumps(SOLUTION),
            "text after": answer(SOLUTION) + " Done.",
            "two blocks": answer(SOLUTION) + answer(SOLUTION),
            "wrong shape": answer(SOLUTION[:3]),
        }.items():
            with self.subTest(name):
                parts = reward_parts(prompt_for(UNIQUE), completion)
                self.assertFalse(parts["format"])
                self.assertIsNone(parts["fill"])
                self.assertEqual(parts["reward"], 0.0)

    def test_ambiguous_puzzle_accepts_any_valid_solution(self):
        prompt = prompt_for(AMBIGUOUS)
        for grid in (SOLUTION, ALTERNATIVE):
            with self.subTest(grid=grid):
                parts = reward_parts(prompt, answer(grid))
                self.assertEqual(parts["reward"], 1.0)
                self.assertFalse(parts["exact"])
                self.assertIsNone(parts["fill"])

    def test_ambiguous_puzzle_gives_no_partial_credit(self):
        invalid = [[1, 1, 3, 4], [3, 4, 1, 2], [1, 2, 4, 3], [4, 3, 2, 1]]
        self.assertEqual(reward(prompt_for(AMBIGUOUS), answer(invalid)), 0.0)

    def test_malformed_prompt_raises(self):
        with self.assertRaises(ValueError):
            reward("not a sudoku prompt", answer(SOLUTION))

    def test_reward_is_one_exactly_when_the_response_passes(self):
        completions = [
            answer(SOLUTION), answer(UNIQUE), answer(with_cell(SOLUTION, 2, 1, 2)),
            answer(with_cell(SOLUTION, 0, 0, 2)), json.dumps(SOLUTION),
            answer(SOLUTION) + " x",
        ]
        for puzzle in (UNIQUE, AMBIGUOUS):
            data = task_from_prompt(prompt_for(puzzle))
            for completion in completions:
                with self.subTest(puzzle=puzzle, completion=completion):
                    report = evaluate_response(completion, data)
                    self.assertEqual(report["reward"] == 1.0, report["passed"])

    def test_reward_stays_in_unit_interval(self):
        for completion in (answer(SOLUTION), answer(UNIQUE), "", answer([[0] * 4] * 4),
                           answer([[9] * 4] * 4)):
            with self.subTest(completion=completion):
                self.assertTrue(0.0 <= reward(prompt_for(UNIQUE), completion) <= 1.0)


class MakeRewardTest(unittest.TestCase):
    def test_custom_weights_change_the_partial_credit_only(self):
        custom = make_reward(RewardWeights(0.2, 0.8))
        prompt = prompt_for(UNIQUE)
        self.assertAlmostEqual(custom(prompt, answer(with_cell(SOLUTION, 2, 1, 2))),
                               0.2 * 2 / 3)
        self.assertEqual(custom(prompt, answer(SOLUTION)), 1.0)
        self.assertEqual(custom(prompt, "no answer"), 0.0)

    def test_exact_one_for_weights_that_do_not_sum_exactly(self):
        custom = make_reward(RewardWeights(0.1, 0.9))
        self.assertEqual(custom(prompt_for(UNIQUE), answer(SOLUTION)), 1.0)


if __name__ == "__main__":
    unittest.main()
