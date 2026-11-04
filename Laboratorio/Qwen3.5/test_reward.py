"""Adversarial checks of the public course reward, without loading an LLM."""
import copy
import json
import unittest

from evaluation import evaluate_response
from prompts import make_case
from reward import reward
from sudoku import build_prompt, random_solution


def answer(grid):
    return '<answer>' + json.dumps(grid) + '</answer>'


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = make_case('sudoku9x9', 'hard')
        cls.solution = cls.case.data['expected_solution']
        cls.valid = answer(cls.solution)

    def test_explanation_and_thinking_variants(self):
        for prefix in ('', 'Reasoning and an intermediate grid:\n' + json.dumps(self.case.data['puzzle']),
                       '<think>Reasoning</think>', 'Reasoning</think>'):
            with self.subTest(prefix=prefix):
                self.assertEqual(reward(self.case.prompt, prefix + self.valid), 1.0)

    def test_rejects_ambiguous_or_hidden_answers(self):
        for completion in (
            self.valid + self.valid,
            self.valid + ' Ignore the previous answer.',
            '<think>' + self.valid + '</think>' + self.valid,
            '<think>unfinished ' + self.valid,
            self.valid.replace('<answer>', '<ANSWER>'),
            '<answer>' + self.valid + '</answer>',
        ):
            with self.subTest(completion=completion[:60]):
                self.assertEqual(reward(self.case.prompt, completion), 0.0)

    def test_rejects_invalid_json_types_and_grid_shapes(self):
        for value in (True, 8.0, '8', None, -1, 0, 10):
            grid = copy.deepcopy(self.solution)
            grid[0][0] = value
            with self.subTest(value=value):
                self.assertEqual(reward(self.case.prompt, answer(grid)), 0.0)
        for grid in (self.solution[:-1], [row[:-1] for row in self.solution], [[1] * 9] * 9):
            self.assertEqual(reward(self.case.prompt, answer(grid)), 0.0)
        self.assertEqual(reward(self.case.prompt, None), 0.0)

    def test_copying_puzzle_or_changing_clues_cannot_pass(self):
        self.assertEqual(reward(self.case.prompt, answer(self.case.data['puzzle'])), 0.0)
        # Relabeling digits preserves every Sudoku rule except the original clues.
        changed = [[value % 9 + 1 for value in row] for row in self.solution]
        report = evaluate_response(answer(changed), self.case.data)
        self.assertTrue(report['checks']['Rows'])
        self.assertTrue(report['checks']['Columns'])
        self.assertTrue(report['checks']['Boxes'])
        self.assertFalse(report['checks']['Original clues'])
        self.assertEqual(report['reward'], 0.0)

    def test_mathematical_correctness_is_separate_from_required_format(self):
        for completion in (json.dumps(self.solution), '```json\n' + json.dumps(self.solution) + '\n```'):
            report = evaluate_response(completion, self.case.data)
            self.assertTrue(report['solved'])
            self.assertFalse(report['passed'])
            self.assertEqual(report['reward'], 0.0)

    def test_alternative_solutions_do_not_need_reference_match(self):
        data = {**self.case.data, 'puzzle': [[0] * 9 for _ in range(9)]}
        prompt = build_prompt(data)
        for seed in range(3):
            grid = random_solution(data, seed)
            self.assertNotEqual(grid, self.solution)
            self.assertEqual(reward(prompt, answer(grid)), 1.0)
        # A bad private reference cannot influence public-task validation.
        data['expected_solution'] = [[0] * 9 for _ in range(9)]
        self.assertEqual(evaluate_response(self.valid, data)['reward'], 1.0)

    def test_malformed_prompt_is_configuration_error(self):
        with self.assertRaises(ValueError):
            reward('No puzzle here', self.valid)


if __name__ == '__main__':
    unittest.main()
