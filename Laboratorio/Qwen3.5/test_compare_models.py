"""Model-free checks: python -m unittest discover -s Laboratorio/Qwen3.5 -p 'test_*.py'."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import compare_models as benchmark
from reward import reward


class ComparisonTests(unittest.TestCase):
    def test_model_registry_reuses_original_2b(self):
        self.assertEqual(benchmark.MODEL_IDS['2b'], 'Qwen/Qwen3.5-2B')
        self.assertEqual(benchmark.MODEL_IDS['9b'], 'mlx-community/Qwen3.5-9B-4bit')
        self.assertNotIn('27b', benchmark.MODEL_IDS)

    def test_tasks_are_reproducible_diverse_and_valid(self):
        tasks = benchmark.make_tasks(3, [10, 20], 42)
        self.assertEqual(tasks, benchmark.make_tasks(3, [10, 20], 42))
        self.assertEqual(len({t['id'] for t in tasks}), 6)
        self.assertEqual(len({json.dumps(t['data']['expected_solution']) for t in tasks}), 6)
        for task in tasks:
            self.assertEqual(sum(v == 0 for row in task['data']['puzzle'] for v in row), task['empty_cells'])
            answer = '<answer>' + json.dumps(task['data']['expected_solution']) + '</answer>'
            self.assertEqual(reward(task['prompt'], answer), 1)

    def test_partial_summary_does_not_hide_missing_tasks(self):
        tasks = benchmark.make_tasks(1, [10, 20], 3)
        row = dict(empty_cells=10, passed=True, solved=True, format_ok=True,
                   token_limit_reached=False, generation_seconds=2)
        summary = benchmark.summarize([row], tasks, 'incomplete')
        self.assertEqual((summary['completed'], summary['planned']), (1, 2))
        self.assertEqual(summary['pass_at_1'], 1)
        self.assertIsNone(summary['by_empty_cells']['20']['pass_at_1'])

    def test_worker_saves_correct_and_invalid_responses(self):
        tasks = benchmark.make_tasks(1, [10, 20], 3)
        answers = iter(['<answer>' + json.dumps(tasks[0]['data']['expected_solution']) + '</answer>', 'bad'])
        def generate(prompt, **kwargs):
            return SimpleNamespace(text=next(answers), prompt_tokens=10, generated_tokens=64,
                                   finish_reason='length')
        models = ModuleType('models')
        models.load_model = lambda config: SimpleNamespace(generate_with_stats=generate)
        mlx = ModuleType('mlx')
        core = ModuleType('mlx.core')
        core.reset_peak_memory = lambda: None
        core.get_peak_memory = lambda: 123
        mlx.core = core
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            benchmark.write_json(output / 'tasks.json', tasks)
            benchmark.write_json(output / 'manifest.json', dict(thinking=True, max_output_tokens=64, max_prompt_tokens=2048, seed=42))
            with patch.dict(sys.modules, {'models': models, 'mlx': mlx, 'mlx.core': core}), patch.object(benchmark, 'version', return_value='test'):
                benchmark.run_worker(output, '9b')
                first = (output / '9b.jsonl').read_text().splitlines()[0]
                (output / '9b.jsonl').write_text(first + '\n')
                answers = iter(['bad'])
                benchmark.run_worker(output, '9b', resume=True)
                # A completed run must not generate any extra response.
                benchmark.run_worker(output, '9b', resume=True)
            rows = [json.loads(line) for line in (output / '9b.jsonl').read_text().splitlines()]
            self.assertEqual([r['passed'] for r in rows], [True, False])
            self.assertTrue(all(r['token_limit_reached'] for r in rows))
            self.assertEqual([r['task_id'] for r in rows], [t['id'] for t in tasks])

    def test_dry_run_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            cmd = [sys.executable, str(Path(benchmark.__file__)), '--dry-run',
                   '--cases-per-level', '1', '--output', str(output)]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(len(json.loads((output / 'tasks.json').read_text())), 3)
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('already exists', second.stderr)


if __name__ == '__main__':
    unittest.main()
