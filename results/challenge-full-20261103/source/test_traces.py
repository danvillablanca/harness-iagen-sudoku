import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from dataset import make_record
from reward import reward
from traces import export_task, generate, load_train, sample_seed


class TraceTests(unittest.TestCase):
    def test_input_rejects_eval_and_duplicates_and_hides_solution(self):
        record = make_record('train', 4, 'easy', 0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'train.jsonl'
            path.write_text(json.dumps(record) + '\n')
            tasks = load_train(path)
            self.assertNotIn('solution', tasks[0])
            self.assertEqual(tasks[0]['prompt'], record['prompt'])
            path.write_text(json.dumps({**record, 'split': 'eval'}))
            with self.assertRaises(ValueError):
                load_train(path)
            path.write_text((json.dumps(record) + '\n') * 2)
            with self.assertRaises(ValueError):
                load_train(path)

    def test_seeds_are_reproducible_and_differ_by_task_and_attempt(self):
        values = [sample_seed(42, task, i) for task in ('a', 'b') for i in range(4)]
        self.assertEqual(len(set(values)), 8)
        self.assertEqual(values[0], sample_seed(42, 'a', 0))
        self.assertTrue(all(0 <= n < 2**32 for n in values))

    def test_no_pairs_without_both_classes_and_exports_are_unique(self):
        task = {'id': 'one', 'prompt': 'p'}
        good = {'completion': 'yes', 'reward': 1., 'candidate_id': 'g'}
        bad = {'completion': 'no', 'reward': 0., 'candidate_id': 'b'}
        self.assertEqual(export_task(task, [good])[1], [])
        self.assertEqual(export_task(task, [bad]), ([], []))
        sft, dpo = export_task(task, [good, good, bad, bad])
        self.assertEqual((len(sft), len(dpo)), (1, 1))
        self.assertEqual(dpo[0]['chosen_id'], 'g')
        self.assertEqual(dpo[0]['rejected_id'], 'b')

    def test_generation_uses_real_reward_and_pairs_only_same_task(self):
        records = [make_record('train', 4, 'easy', i) for i in range(2)]
        answers = {r['prompt']: '<answer>' + json.dumps(r['solution']) + '</answer>' for r in records}
        counts, calls = {}, []
        def respond(prompt, **kwargs):
            calls.append(kwargs)
            counts[prompt] = counts.get(prompt, 0) + 1
            text = answers[prompt] if counts[prompt] == 2 else 'invalid'
            return SimpleNamespace(text=text, prompt_tokens=10, generated_tokens=20, finish_reason='stop')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            summary = generate(records, SimpleNamespace(generate_with_stats=respond), output,
                               k=3, temperature=.7, top_p=.95, seed=1, model_id='fake', max_tokens=100)
            sft = [json.loads(x) for x in (output / 'sft.jsonl').read_text().splitlines()]
            pairs = [json.loads(x) for x in (output / 'dpo.jsonl').read_text().splitlines()]
            raw = [json.loads(x) for x in (output / 'candidates.jsonl').read_text().splitlines()]
            self.assertEqual((summary['status'], len(raw), len(sft), len(pairs)), ('complete', 6, 2, 2))
            for row in pairs:
                self.assertEqual(reward(row['prompt'], row['chosen']), 1.)
                self.assertEqual(reward(row['prompt'], row['rejected']), 0.)
                self.assertTrue(row['chosen_id'].startswith(row['task_id'] + ':'))
                self.assertTrue(row['rejected_id'].startswith(row['task_id'] + ':'))
            self.assertTrue(all(call['temperature'] == .7 for call in calls))
            self.assertEqual(len({call['seed'] for call in calls}), 6)


if __name__ == '__main__':
    unittest.main()
