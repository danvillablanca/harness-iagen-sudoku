"""Tests for the SFT/DPO dataset builder. Run from the repository root:

    python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"
"""

import unittest

from build_sft_dpo import dpo_pairs, group_by_task, sft_examples


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


if __name__ == "__main__":
    unittest.main()
