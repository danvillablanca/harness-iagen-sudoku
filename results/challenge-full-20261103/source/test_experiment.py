import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from baseline import pending, run, summarize
from dataset import make_record, response_style, save
from experiment import digest, load_samples, prepare_manifest
from traces import generate
from test_baseline import SETTINGS, sample


class PersistenceTests(unittest.TestCase):
    def test_resume_recovers_only_torn_tail_and_preserves_it(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'samples.jsonl'
            path.write_bytes(b'{"a": 1}\n{"a":')
            with self.assertRaises(ValueError):
                load_samples(path)
            self.assertEqual(load_samples(path, repair_tail=True), [{'a': 1}])
            self.assertEqual(path.with_suffix('.jsonl.partial').read_bytes(), b'{"a":')
            path.write_bytes(b'{"a": 1}\nBAD\n')
            with self.assertRaises(ValueError):
                load_samples(path, repair_tail=True)

    def test_resume_rejects_changed_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            original = prepare_manifest(path, {'temperature': .7})
            self.assertEqual(prepare_manifest(path, {'temperature': .7}, resume=True), original)
            with self.assertRaises(ValueError):
                prepare_manifest(path, {'temperature': 1.}, resume=True)

    def test_dataset_final_style_keeps_puzzles_and_refuses_overwrite(self):
        record = make_record('train', 4, 'easy', 0)
        dataset = response_style({'train': [dict(record)]}, 'final')
        self.assertEqual(record['puzzle'], dataset['train'][0]['puzzle'])
        self.assertNotEqual(record['prompt'], dataset['train'][0]['prompt'])
        with tempfile.TemporaryDirectory() as temp:
            save(dataset, temp)
            with self.assertRaises(ValueError):
                save(dataset, temp)


class MetricsIntegrityTests(unittest.TestCase):
    def test_duplicate_samples_are_rejected_not_counted_twice(self):
        a = sample('a', 0, 1.)
        with self.assertRaises(ValueError):
            summarize([a, a])
        with self.assertRaises(ValueError):
            pending([{'id': 'a'}], [a, a], SETTINGS)

    def test_distinct_decoding_runs_are_not_pooled(self):
        rows = [sample('a', i, 1.) for i in range(4)]
        other = [{**sample('a', i, 0.), 'temperature': 1.} for i in range(4)]
        values = summarize(rows + other, ks=(1, 4))
        self.assertEqual(len(values), 2)
        self.assertEqual(sorted(r['pass@1'] for r in values), [0., 1.])

    def test_missing_tasks_are_visible_and_incomplete_tasks_excluded(self):
        manifest = {'run_id': 'r', 'settings': SETTINGS, 'tasks': [
            {'id': name, 'size': 4, 'level': 'easy', 'split': 'eval'} for name in ('a', 'b')]}
        rows = [{**sample('a', i, 1.), 'run_id': 'r', 'split': 'eval'} for i in range(4)]
        result = summarize(rows, ks=(1, 4, 8), planned=[manifest])[0]
        self.assertEqual((result['tasks'], result['tasks_planned']), (1, 2))
        self.assertTrue(result['provisional'])
        self.assertEqual(result['tasks@4'], 1)
        self.assertIsNone(result['pass@8'])

    def test_mismatched_metadata_is_rejected(self):
        rows = [{**sample('a', i, 1.), 'run_id': 'r'} for i in range(4)]
        rows[1]['size'] = 9
        with self.assertRaises(ValueError):
            summarize(rows)


class ResumeIntegrationTests(unittest.TestCase):
    def test_teacher_resume_preserves_samples_exports_and_seeds(self):
        task = make_record('train', 4, 'easy', 0)
        correct = '<answer>' + json.dumps(task['solution']) + '</answer>'
        def response(text):
            return SimpleNamespace(text=text, prompt_tokens=30, generated_tokens=20, finish_reason='stop')
        calls = []
        def interrupted(prompt, **kwargs):
            calls.append(kwargs['seed'])
            if len(calls) == 2:
                raise KeyboardInterrupt
            return response(correct)
        options = dict(k=3, temperature=.7, top_p=.8, seed=42, model_id='fake', max_tokens=64)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with self.assertRaises(KeyboardInterrupt):
                generate([task], SimpleNamespace(generate_with_stats=interrupted), out, **options)
            first = (out / 'candidates.jsonl').read_text()
            self.assertEqual(len(load_samples(out / 'sft.jsonl')), 1)
            resumed_seeds = []
            def resumed(prompt, **kwargs):
                resumed_seeds.append(kwargs['seed'])
                return response('invalid')
            result = generate([task], SimpleNamespace(generate_with_stats=resumed), out, resume=True, **options)
            self.assertTrue((out / 'candidates.jsonl').read_text().startswith(first))
            self.assertEqual(resumed_seeds[0], calls[1])
            self.assertEqual((result['samples'], result['sft_records'], result['dpo_pairs']), (3, 1, 1))
            # A completed resume must not need a model at all.
            self.assertEqual(generate([task], None, out, resume=True, **options), result)

    def test_baseline_end_to_end_then_resume_without_loading_model(self):
        task = make_record('eval', 4, 'easy', 0)
        correct = '<answer>' + json.dumps(task['solution']) + '</answer>'
        calls = []
        def generate_response(prompt, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text=correct if len(calls) % 2 else 'invalid',
                                   prompt_tokens=10, generated_tokens=20, finish_reason='stop')
        model = SimpleNamespace(backend='fake', generate_with_stats=generate_response)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'eval.jsonl'
            path.write_text(json.dumps(task) + '\n')
            args = SimpleNamespace(tasks=path, model_id='fake', backend='cpu', thinking=False,
                                   n=4, temperature=.7, top_p=.8, top_k=20, presence_penalty=1.5,
                                   seed=42, max_output_tokens=64, max_prompt_tokens=2048, answer_prefix=False,
                                   per_level=None, sizes=None, out=Path(temp)/'out', resume=False, dry_run=False)
            with patch('baseline.provenance', return_value={}), patch('models.load_model', return_value=model) as loader:
                self.assertEqual(run(args), 0)
                args.resume = True
                self.assertEqual(run(args), 0)
                self.assertEqual(loader.call_count, 1)
            result = json.loads(next(args.out.glob('*.summary.json')).read_text())
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['metrics'][0]['pass@1'], .5)
            self.assertEqual(result['metrics'][0]['pass@4'], 1.)
            self.assertEqual(len({r['seed'] for r in calls}), 4)


if __name__ == '__main__':
    unittest.main()
