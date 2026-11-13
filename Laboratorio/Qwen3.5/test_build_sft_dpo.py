"""Tests for the SFT/DPO dataset builder. Run from the repository root:

    python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"
"""

import random
import unittest
from collections import Counter

from build_sft_dpo import (
    corrupt_answer, dpo_pairs, group_by_task, is_wrong_grid, sft_examples, synthetic_pairs,
)
from dataset import make_record
from reward import reward


def sample(index, reward, tokens=10, finish="stop", task="t", split="train", text=None):
    return {"task_id": task, "split": split, "size": 4, "level": "easy", "sample": index,
            "reward": reward, "finish_reason": finish, "generated_tokens": tokens,
            "completion": text or f"completion {task} {index}"}


PROMPTS = {"t": "prompt t", "u": "prompt u"}


class GroupingTest(unittest.TestCase):
    def test_groups_by_task(self):
        groups = group_by_task([sample(0, 1.0), sample(1, 0.0), sample(0, 1.0, task="u")])
        self.assertEqual({k: len(v) for k, v in groups.items()}, {"t": 2, "u": 1})

    def test_refuses_other_splits(self):
        with self.assertRaises(ValueError):
            group_by_task([sample(0, 1.0, split="eval")])


class SftTest(unittest.TestCase):
    def test_keeps_shortest_distinct_correct_samples(self):
        groups = group_by_task([
            sample(0, 1.0, tokens=30), sample(1, 1.0, tokens=10),
            sample(2, 1.0, tokens=20), sample(3, 1.0, tokens=5, text="completion t 1"),
            sample(4, 0.0, tokens=1),
        ])
        examples = sft_examples(groups, PROMPTS, max_per_task=2)
        # Sample 3 is the shortest; sample 1 repeats its text, so sample 2 comes next.
        self.assertEqual([e["completion"] for e in examples],
                         ["completion t 1", "completion t 2"])
        self.assertTrue(all(e["prompt"] == "prompt t" for e in examples))

    def test_skips_tasks_without_correct_samples(self):
        groups = group_by_task([sample(0, 0.0), sample(1, 0.0, finish="length")])
        self.assertEqual(sft_examples(groups, PROMPTS, 2), [])


class DpoTest(unittest.TestCase):
    def test_prefers_wrong_grids_over_truncated_and_near_misses_first(self):
        groups = group_by_task([
            sample(0, 1.0, tokens=10), sample(1, 1.0, tokens=20),
            sample(2, 0.0, finish="length"), sample(3, 0.0), sample(4, 0.4),
        ])
        pairs = dpo_pairs(groups, PROMPTS, max_per_task=2)
        self.assertEqual([p["rejected"] for p in pairs],
                         ["completion t 4", "completion t 3"])
        self.assertEqual([p["chosen"] for p in pairs],
                         ["completion t 0", "completion t 1"])
        self.assertFalse(any(p["rejected_truncated"] for p in pairs))

    def test_uses_truncated_when_no_wrong_grid_and_marks_it(self):
        groups = group_by_task([sample(0, 1.0), sample(1, 0.0, finish="length")])
        (pair,) = dpo_pairs(groups, PROMPTS, max_per_task=3)
        self.assertTrue(pair["rejected_truncated"])

    def test_pairs_limited_by_cap_and_by_either_side(self):
        groups = group_by_task([sample(i, 1.0) for i in range(5)] + [sample(5, 0.0)])
        self.assertEqual(len(dpo_pairs(groups, PROMPTS, max_per_task=3)), 1)
        groups = group_by_task([sample(i, 1.0) for i in range(3)]
                               + [sample(i, 0.0) for i in range(3, 6)])
        self.assertEqual(len(dpo_pairs(groups, PROMPTS, max_per_task=2)), 2)

    def test_no_pair_when_all_correct_or_all_failed(self):
        self.assertEqual(dpo_pairs(group_by_task([sample(0, 1.0)]), PROMPTS, 2), [])
        self.assertEqual(dpo_pairs(group_by_task([sample(0, 0.0)]), PROMPTS, 2), [])


class SyntheticTest(unittest.TestCase):
    record = make_record("train", 4, "easy", 0)

    def completion(self, text="Reasoning."):
        return f"{text}\n\n<answer>{self.record['solution']}</answer>".replace("'", "")

    def test_corrupts_one_blank_cell_and_breaks_the_solution(self):
        good = self.completion()
        self.assertEqual(reward(self.record["prompt"], good), 1.0)
        bad = corrupt_answer(good, self.record["puzzle"], random.Random(0))
        self.assertTrue(bad.startswith("Reasoning."))
        self.assertTrue(is_wrong_grid(self.record["prompt"], bad))
        old = eval(good.split("<answer>")[1].split("</answer>")[0])
        new = eval(bad.split("<answer>")[1].split("</answer>")[0])
        changed = [(r, c) for r in range(4) for c in range(4) if old[r][c] != new[r][c]]
        self.assertEqual(len(changed), 1)
        r, c = changed[0]
        self.assertEqual(self.record["puzzle"][r][c], 0)

    def test_is_deterministic_and_needs_an_answer(self):
        good = self.completion()
        puzzle = self.record["puzzle"]
        self.assertEqual(corrupt_answer(good, puzzle, random.Random(1)),
                         corrupt_answer(good, puzzle, random.Random(1)))
        self.assertIsNone(corrupt_answer("no answer here", puzzle, random.Random(0)))
        self.assertIsNone(corrupt_answer("<answer>nope</answer>", puzzle, random.Random(0)))

    def test_synthetic_pairs_rotate_chosen_and_are_marked(self):
        task = self.record["id"]
        groups = group_by_task([
            sample(0, 1.0, tokens=5, task=task, text=self.completion("short")),
            sample(1, 1.0, tokens=9, task=task, text=self.completion("longer")),
        ])
        prompts = {task: self.record["prompt"]}
        puzzles = {task: self.record["puzzle"]}
        pairs = synthetic_pairs(groups, prompts, puzzles, 2, Counter({task: 1}))
        self.assertEqual(len(pairs), 2)
        self.assertEqual([p["chosen"].split("\n")[0] for p in pairs], ["longer", "short"])
        self.assertTrue(all(p["rejected_synthetic"] and not p["rejected_truncated"]
                            for p in pairs))
        self.assertNotEqual(pairs[0]["chosen"], pairs[0]["rejected"])

    def test_no_synthetic_pairs_without_correct_samples(self):
        groups = group_by_task([sample(0, 0.0, task=self.record["id"])])
        self.assertEqual(synthetic_pairs(groups, {}, {}, 2), [])


if __name__ == "__main__":
    unittest.main()
