"""Tests for dataset generation. Run from the repository root:

    python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"
"""

import json
import random
import unittest

from dataset import LEVELS, TEST_CELLS, blank_cells, board, case_from_record, generate, make_record
from evaluation import evaluate
from reward import reward
from sudoku import count_solutions, evaluate_grid, random_solution, task_from_prompt

COUNTS = {"train": 3, "eval": 2, "test": 2}


def answer(grid):
    return f"<answer>{json.dumps(grid)}</answer>"


class CountSolutionsTest(unittest.TestCase):
    def test_complete_grid_has_one_solution(self):
        case = board(9)
        solution = random_solution(case, seed=1)
        self.assertEqual(count_solutions(solution, case), 1)

    def test_empty_grid_stops_at_limit(self):
        case = board(4)
        self.assertEqual(count_solutions([[0] * 4 for _ in range(4)], case), 2)
        self.assertEqual(count_solutions([[0] * 4 for _ in range(4)], case, limit=5), 5)

    def test_blank_cells_reports_impossible_unique_targets(self):
        case = board(4)
        solution = random_solution(case, seed=1)
        # A 4x4 puzzle with a unique solution needs at least 4 clues.
        self.assertIsNone(blank_cells(solution, case, 16, random.Random(0)))


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate(COUNTS, seed=7)

    def records(self):
        return [record for records in self.dataset.values() for record in records]

    def test_counts_per_split(self):
        levels = sum(len(LEVELS[size]) for size in (4, 6, 9))
        for split, count in COUNTS.items():
            self.assertEqual(len(self.dataset[split]), count * levels)

    def test_deterministic(self):
        self.assertEqual(generate(COUNTS, seed=7), self.dataset)
        self.assertNotEqual(generate(COUNTS, seed=8), self.dataset)

    def test_more_training_tasks_keep_eval_and_test(self):
        bigger = generate({**COUNTS, "train": 5}, seed=7)
        self.assertEqual(bigger["test"], self.dataset["test"])
        self.assertEqual(bigger["eval"], self.dataset["eval"])

    def test_splits_share_no_puzzle(self):
        keys = [(r["size"], json.dumps(r["puzzle"])) for r in self.records()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_records_are_consistent(self):
        for record in self.records():
            with self.subTest(record["id"]):
                case = board(record["size"])
                puzzle, solution = record["puzzle"], record["solution"]
                checks = evaluate_grid(solution, {**case, "puzzle": puzzle})["checks"]
                self.assertTrue(all(checks.values()))
                empty = sum(value == 0 for row in puzzle for value in row)
                self.assertEqual(empty, record["empty_cells"])
                low, high = LEVELS[record["size"]][record["level"]]
                self.assertTrue(low <= empty <= high)
                self.assertTrue(record["unique_solution"])
                self.assertEqual(count_solutions(puzzle, case), 1)
                self.assertEqual(task_from_prompt(record["prompt"])["puzzle"], puzzle)

    def test_reward_accepts_solution_and_rejects_puzzle(self):
        for record in self.records():
            with self.subTest(record["id"]):
                self.assertEqual(reward(record["prompt"], answer(record["solution"])), 1.0)
                self.assertEqual(reward(record["prompt"], answer(record["puzzle"])), 0.0)

    def test_case_from_record_evaluates(self):
        record = self.dataset["test"][0]
        report = evaluate(answer(record["solution"]), case_from_record(record))
        self.assertTrue(report["passed"])

    def test_cell_grid_gives_the_test_split_fixed_blank_counts(self):
        grid = generate(COUNTS, seed=7, test_grid="cells")
        counts = {(r["size"], r["level"]): set() for r in grid["test"]}
        for record in grid["test"]:
            counts[record["size"], record["level"]].add(record["empty_cells"])
        expected = {(size, f"n{cells}"): {cells}
                    for size, cells_list in TEST_CELLS.items() for cells in cells_list}
        self.assertEqual(counts, expected)
        self.assertEqual(len(grid["test"]), COUNTS["test"] * len(expected))
        # Eval and train keep the difficulty levels, and nothing is shared.
        self.assertEqual({r["level"] for r in grid["eval"] + grid["train"]},
                         {"easy", "medium", "hard"})
        keys = [(r["size"], json.dumps(r["puzzle"]))
                for records in grid.values() for r in records]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(r["unique_solution"] for r in grid["test"]))

    def test_cell_grid_does_not_change_the_default(self):
        self.assertEqual(generate(COUNTS, seed=7, test_grid="levels"), self.dataset)
        with self.assertRaises(ValueError):
            generate(COUNTS, test_grid="other")

    def test_without_uniqueness_reaches_requested_blanks(self):
        record = make_record("train", 4, "hard", 0, seed=7, unique=False)
        low, high = LEVELS[4]["hard"]
        self.assertTrue(low <= record["empty_cells"] <= high)


if __name__ == "__main__":
    unittest.main()
