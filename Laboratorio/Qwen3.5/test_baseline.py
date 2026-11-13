"""Tests for the baseline metrics and batched sampling. Run from the repository root:

    python -m unittest discover -s Laboratorio/Qwen3.5 -p "test_*.py"
"""

import json
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from baseline import (
    RUN_KEYS, format_table, output_path, parse_shard, parse_skip, pass_at_k, pending, run, select_tasks,
    summarize, take_shard,
)
from config import ModelConfig
from dataset import make_record

SETTINGS = {"model": "m", "thinking": False, "temperature": 0.7, "top_p": 0.8,
            "max_tokens": 64, "n": 4, "seed": 0, "shots": 0, "shot_seed": 0,
            "shot_format": "steps"}


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

    def test_skip_leaves_out_size_level_pairs(self):
        chosen = select_tasks(self.records, skip=parse_skip(["4:hard", "9:easy"]))
        self.assertEqual([r["id"] for r in chosen], ["t0", "t1", "t2"])

    def test_shards_partition_tasks_and_keep_the_mix(self):
        shards = [take_shard(self.records, parse_shard(f"{i}/2")) for i in range(2)]
        ids = sorted(r["id"] for shard in shards for r in shard)
        self.assertEqual(ids, sorted(r["id"] for r in self.records))
        self.assertEqual(take_shard(self.records, None), self.records)

    def test_parse_shard_rejects_malformed_values(self):
        for value in ("5", "5/5", "a/2", "1/", "-1/2"):
            with self.subTest(value):
                with self.assertRaises(ValueError):
                    parse_shard(value)

    def test_parse_skip_rejects_malformed_values(self):
        for value in ("hard", "6:", ":hard", "x:hard"):
            with self.subTest(value):
                with self.assertRaises(ValueError):
                    parse_skip([value])


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


class LegacyResultsTest(unittest.TestCase):
    """Result files written before shots existed carry no shots keys."""
    tasks = [{"id": "a"}]

    def legacy(self, index):
        done = {k: v for k, v in sample("a", index, 1.0).items()
                if k not in ("shots", "shot_seed", "shot_format")}
        return done

    def test_legacy_samples_count_as_zero_shots(self):
        self.assertEqual(pending(self.tasks, [self.legacy(0)], SETTINGS),
                         [(self.tasks[0], [1, 2, 3])])

    def test_legacy_samples_are_rejected_when_shots_are_requested(self):
        with self.assertRaises(ValueError):
            pending(self.tasks, [self.legacy(0)], {**SETTINGS, "shots": 2})

    def test_output_path_separates_shot_counts(self):
        zero = output_path("out", "Qwen/Qwen3.5-2B", False)
        self.assertEqual(zero.name, "Qwen__Qwen3.5-2B-nothinking.jsonl")
        self.assertEqual(output_path("out", "Qwen/Qwen3.5-2B", True, 3).name,
                         "Qwen__Qwen3.5-2B-thinking-3shot.jsonl")
        self.assertEqual(output_path("out", "m", False, 2, "compact").name,
                         "m-nothinking-2shot-compact.jsonl")
        # The format only matters, and only names the file, when shots are used.
        self.assertEqual(output_path("out", "m", False, 0, "compact").name,
                         "m-nothinking.jsonl")


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
        # Only 2 samples: pass@4 cannot be estimated.
        self.assertEqual(rows[9, "hard"]["pass@1"], 1.0)
        self.assertIsNone(rows[9, "hard"]["pass@4"])

    def test_shot_counts_are_separate_rows(self):
        samples = (
            [sample("a", i, 0.0) for i in range(2)]
            + [{**sample("a", i, 1.0), "shots": 2} for i in range(2)]
        )
        samples += [{**sample("a", i, 0.5 if i else 1.0), "shots": 2, "shot_format": "compact"}
                    for i in range(2)]
        rows = {r["shots"]: r for r in summarize(samples, ks=(1,))}
        self.assertEqual(rows[0]["pass@1"], 0.0)
        self.assertEqual(rows[2]["pass@1"], 1.0)
        self.assertEqual(rows["2 (compact)"]["pass@1"], 0.5)
        self.assertIn("shots", format_table(list(rows.values()), ks=(1,)))

    def test_table_shows_blank_cells_and_orders_by_difficulty(self):
        samples = (
            [{**sample("a", i, 1.0, size=9, level="n30"), "empty_cells": 30} for i in range(2)]
            + [{**sample("b", i, 0.0, size=9, level="n10"), "empty_cells": 10} for i in range(2)]
        )
        rows = {r["level"]: r for r in summarize(samples, ks=(1,))}
        self.assertEqual(rows["n30"]["empty_cells"], 30)
        lines = format_table(list(rows.values()), ks=(1,)).splitlines()
        self.assertIn("empty", lines[0])
        self.assertIn("n10", lines[2])
        self.assertIn("n30", lines[3])


def answer(grid):
    return f"<answer>{json.dumps(grid)}</answer>"


class RunTest(unittest.TestCase):
    """`run` end to end with a fake model: shots, stored traces and resuming."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        seen = set()
        self.records = [make_record("eval", 4, "easy", i, 0, True, seen) for i in range(2)]
        self.tasks = Path(self.tmp.name, "tasks.jsonl")
        self.tasks.write_text("".join(json.dumps(r) + "\n" for r in self.records))
        self.calls, self.configs = [], []

        def generate_samples(prompt, n, **kwargs):
            self.calls.append({"prompt": prompt, "n": n, **kwargs})
            record = next(r for r in self.records if r["prompt"] == prompt)
            grids = [record["solution"], record["puzzle"]]
            return [SimpleNamespace(text=answer(grids[i % 2]), finish_reason="stop",
                                    generated_tokens=7) for i in range(n)]

        def load_model(config):
            self.configs.append(config)
            return SimpleNamespace(backend="fake", generate_samples=generate_samples)

        self.stub = SimpleNamespace(load_model=load_model)

    def args(self, **overrides):
        values = dict(tasks=str(self.tasks), model_id="Qwen/Qwen3.5-2B", backend="cpu",
                      thinking=False, n=2, batch_size=2, max_output_tokens=64,
                      max_prompt_tokens=None, temperature=None, top_p=None,
                      per_level=None, sizes=None, skip=None, shard=None, seed=0, out=self.tmp.name,
                      shots=0, shot_seed=0, shot_format="steps")
        return SimpleNamespace(**{**values, **overrides})

    def run_with_stub(self, **overrides):
        with mock.patch.dict(sys.modules, {"models": self.stub}):
            return run(self.args(**overrides))

    def test_stores_scored_traces_and_resumes(self):
        self.assertEqual(self.run_with_stub(), 0)
        path = output_path(self.tmp.name, "Qwen/Qwen3.5-2B", False)
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(lines), 4)
        for line in lines:
            self.assertEqual(line["shots"], 0)
            self.assertIn("<answer>", line["completion"])
            self.assertEqual(set(line["reward_parts"]),
                             {"format", "clues_frac", "fill", "solved", "exact"})
        correct = [line for line in lines if line["sample"] == 0]
        wrong = [line for line in lines if line["sample"] == 1]
        self.assertTrue(all(line["reward"] == 1.0 and line["reward_parts"]["solved"]
                            for line in correct))
        self.assertTrue(all(line["reward"] == 0.0 for line in wrong))
        self.assertEqual(self.configs[0].max_prompt_tokens, 2048)

        # Same arguments again: nothing pending, so the model is not even loaded.
        self.assertEqual(self.run_with_stub(), 0)
        self.assertEqual(len(self.configs), 1)
        self.assertEqual(len(path.read_text().splitlines()), 4)

    def test_resumes_after_an_interruption(self):
        self.run_with_stub()
        path = output_path(self.tmp.name, "Qwen/Qwen3.5-2B", False)
        kept = path.read_text().splitlines()[:3]
        path.write_text("\n".join(kept) + "\n")
        self.calls.clear()
        self.run_with_stub()
        self.assertEqual(sum(call["n"] for call in self.calls), 1)
        self.assertEqual(len(path.read_text().splitlines()), 4)

    def test_shots_are_passed_to_the_model_and_recorded(self):
        self.run_with_stub(shots=2, shot_seed=3, shot_format="compact")
        path = output_path(self.tmp.name, "Qwen/Qwen3.5-2B", False, 2, "compact")
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertTrue(all(line["shots"] == 2 and line["shot_seed"] == 3
                            and line["shot_format"] == "compact" for line in lines))
        self.assertEqual(self.configs[0].max_prompt_tokens, 16384)
        for call in self.calls:
            self.assertEqual(len(call["shots"]), 2)
            for user, assistant in call["shots"]:
                self.assertNotEqual(user, call["prompt"])
                self.assertIn("<answer>", assistant)
                self.assertRegex(assistant, r"(?m)^r\dc\d=\d$")


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
