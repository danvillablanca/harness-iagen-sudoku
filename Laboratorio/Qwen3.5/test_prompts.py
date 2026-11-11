"""Tests for the worked-example (shot) formats. Run from the repository root:

    python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"
"""

import re
import unittest

from dataset import board, make_record
from evaluation import evaluate_response
from prompts import SHOT_FORMATS, _worked_example, make_shots
from sudoku import task_from_prompt


def worked_case(size, empty_cells=8, seed=3):
    record = make_record("train", size, "easy", seed, 0, True, None,
                         {"easy": (empty_cells, empty_cells)})
    case = {**board(size), "puzzle": record["puzzle"], "expected_solution": record["solution"]}
    return record, case


class ShotFormatTest(unittest.TestCase):
    def test_every_format_ends_in_a_solving_answer(self):
        # The reply to a shot's own prompt must satisfy the response contract.
        for style in SHOT_FORMATS:
            for size in (4, 6, 9):
                with self.subTest(style=style, size=size):
                    record, case = worked_case(size)
                    reply = _worked_example(case, style)
                    report = evaluate_response(reply, task_from_prompt(record["prompt"]))
                    self.assertTrue(report["passed"])
                    self.assertEqual(report["reward"], 1.0)

    def test_steps_is_the_default_format(self):
        _, case = worked_case(4)
        self.assertEqual(_worked_example(case), _worked_example(case, "steps"))
        self.assertIn("Step 1:", _worked_example(case))

    def test_compact_has_one_short_line_per_blank(self):
        _, case = worked_case(6, empty_cells=10)
        lines = _worked_example(case, "compact").splitlines()
        self.assertEqual(len(lines), 10 + 1)
        self.assertTrue(all(re.fullmatch(r"r\dc\d=\d", line) for line in lines[:-1]))
        self.assertTrue(lines[-1].startswith("<answer>"))

    def test_seen_lists_every_other_digit_and_the_chosen_one_is_missing(self):
        _, case = worked_case(9, empty_cells=12)
        lines = _worked_example(case, "seen").splitlines()
        self.assertEqual(len(lines), 12 + 1)
        for line in lines[:-1]:
            match = re.fullmatch(r"r\dc\d sees ([\d,]+) -> (\d)", line)
            self.assertIsNotNone(match, line)
            seen = match.group(1).split(",")
            self.assertEqual(len(seen), 8)
            self.assertNotIn(match.group(2), seen)

    def test_compact_formats_are_shorter_than_steps(self):
        _, case = worked_case(9, empty_cells=20)
        sizes = {style: len(_worked_example(case, style)) for style in SHOT_FORMATS}
        self.assertLess(sizes["compact"], sizes["seen"])
        self.assertLess(sizes["seen"], sizes["steps"])

    def test_make_shots_uses_the_requested_format(self):
        record, _ = worked_case(4)
        shots = make_shots(board(4), record["puzzle"], 2, 0, "compact")
        self.assertEqual(len(shots), 2)
        self.assertTrue(all(re.search(r"^r\dc\d=\d$", reply, re.M) for _, reply in shots))

    def test_unknown_format_is_rejected(self):
        _, case = worked_case(4)
        with self.assertRaises(ValueError):
            _worked_example(case, "verbose")
        with self.assertRaises(ValueError):
            make_shots(board(4), case["puzzle"], 1, 0, "verbose")


if __name__ == "__main__":
    unittest.main()
