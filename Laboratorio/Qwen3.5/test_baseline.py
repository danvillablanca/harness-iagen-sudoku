"""Tests for the baseline metrics and batched sampling. Run from the repository root:

    python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"
"""

import unittest
from importlib.util import find_spec
from types import SimpleNamespace

from baseline import RUN_KEYS, pass_at_k, pending, select_tasks, summarize
from config import ModelConfig

SETTINGS = {"model": "m", "thinking": False, "temperature": 0.7, "top_p": 0.8,
            "max_tokens": 64, "n": 4, "seed": 0, "top_k": 20, "presence_penalty": 1.5,
            "backend": "cpu", "max_prompt_tokens": 2048, "answer_prefix": False}


def sample(task_id, index, reward, size=4, level="easy", finish="stop", tokens=10):
    return {**SETTINGS, "task_id": task_id, "size": size, "level": level,
            "sample": index, "reward": reward, "finish_reason": finish,
            "generated_tokens": tokens}


class PassAtKTest(unittest.TestCase):
    def test_extremes(self):
        self.assertEqual(pass_at_k(8, 0, 4), 0.0)
        self.assertEqual(pass_at_k(8, 8, 1), 1.0)

    def test_pass_at_1_is_success_rate(self):
        self.assertAlmostEqual(pass_at_k(8, 3, 1), 3 / 8)

    def test_known_value(self):
        # 1 - C(3, 2) / C(5, 2) = 1 - 3/10
        self.assertAlmostEqual(pass_at_k(5, 2, 2), 0.7)

    def test_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            pass_at_k(4, 5, 1)
        with self.assertRaises(ValueError):
            pass_at_k(4, 1, 5)


class SelectionTest(unittest.TestCase):
    records = [{"id": f"t{i}", "size": size, "level": level}
               for i, (size, level) in enumerate(
                   [(4, "easy")] * 3 + [(4, "hard")] * 3 + [(9, "easy")] * 3)]

    def test_per_level_and_sizes(self):
        chosen = select_tasks(self.records, per_level=2, sizes=[4])
        self.assertEqual([r["id"] for r in chosen], ["t0", "t1", "t3", "t4"])

    def test_all_by_default(self):
        self.assertEqual(select_tasks(self.records), self.records)


class ResumeTest(unittest.TestCase):
    tasks = [{"id": "a"}, {"id": "b"}]

    def test_skips_finished_samples(self):
        done = [sample("a", i, 1.0) for i in range(4)] + [sample("b", 0, 0.0)]
        self.assertEqual(pending(self.tasks, done, SETTINGS), [(self.tasks[1], [1, 2, 3])])

    def test_rejects_different_settings(self):
        for key in RUN_KEYS:
            with self.subTest(key):
                done = [{**sample("a", 0, 1.0), key: "other"}]
                with self.assertRaises(ValueError):
                    pending(self.tasks, done, SETTINGS)


class SummaryTest(unittest.TestCase):
    def test_groups_and_metrics(self):
        samples = (
            [sample("a", i, 1.0 if i == 0 else 0.0) for i in range(4)]
            + [sample("b", i, 0.0, finish="length", tokens=64) for i in range(4)]
            + [sample("c", i, 1.0, size=9, level="hard") for i in range(2)]
        )
        rows = {(r["size"], r["level"]): r for r in summarize(samples, ks=(1, 4))}
        easy = rows[4, "easy"]
        self.assertEqual(easy["tasks"], 2)
        self.assertAlmostEqual(easy["pass@1"], (0.25 + 0.0) / 2)
        self.assertAlmostEqual(easy["pass@4"], (1.0 + 0.0) / 2)
        self.assertAlmostEqual(easy["truncated"], 0.5)
        self.assertAlmostEqual(easy["avg_tokens"], 37.0)
        # Only 2 of the planned 4 samples: this task is incomplete, not scored yet.
        self.assertIsNone(rows[9, "hard"]["pass@1"])
        self.assertTrue(rows[9, "hard"]["provisional"])
        self.assertEqual(rows[9, "hard"]["tasks"], 0)
        self.assertIsNone(rows[9, "hard"]["pass@4"])


@unittest.skipUnless(find_spec("torch") and find_spec("psutil"), "requires torch")
class GenerateSamplesTest(unittest.TestCase):
    EOS = 2
    PROMPT = [1, 1, 1]

    def model_returning(self, continuations):
        import torch

        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return torch.tensor([self.PROMPT + row for row in continuations])

        model = SimpleNamespace(generate=generate,
                                generation_config=SimpleNamespace(eos_token_id=[self.EOS]))
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda *a, **k: list(self.PROMPT),
            decode=lambda ids, skip_special_tokens: " ".join(map(str, ids)),
            all_special_tokens=[], pad_token_id=self.EOS, eos_token_id=self.EOS,
        )
        from qwen import Qwen

        return Qwen(model, tokenizer, "cpu", ModelConfig(max_output_tokens=5)), calls

    def test_trims_padding_and_reports_finish_reason(self):
        qwen, calls = self.model_returning([[5, 6, 2, 2, 2], [5, 6, 7, 8, 9]])
        stopped, truncated = qwen.generate_samples("p", 2, temperature=0.7, seed=1)
        self.assertEqual((stopped.text, stopped.generated_tokens, stopped.finish_reason),
                         ("5 6 2", 3, "stop"))
        self.assertEqual((truncated.generated_tokens, truncated.finish_reason), (5, "length"))
        self.assertTrue(calls[0]["do_sample"])
        self.assertEqual(calls[0]["num_return_sequences"], 2)

    def test_greedy_generation_is_unchanged(self):
        qwen, calls = self.model_returning([[5, 6, 2]])
        result = qwen.generate_with_stats("p")
        self.assertEqual((result.text, result.finish_reason, result.prompt_tokens),
                         ("5 6 2", "stop", 3))
        self.assertFalse(calls[0]["do_sample"])

    def test_rejects_greedy_temperature(self):
        qwen, _ = self.model_returning([[2]])
        with self.assertRaises(ValueError):
            qwen.generate_samples("p", 2, temperature=0)


if __name__ == "__main__":
    unittest.main()
