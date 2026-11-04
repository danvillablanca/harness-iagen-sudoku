"""Check generation arguments at both runtime boundaries without downloading models."""
from types import SimpleNamespace
from importlib.util import find_spec
import unittest
from unittest.mock import Mock, patch

from config import ModelConfig
from qwen import Qwen


class SamplingTests(unittest.TestCase):
    def make_model(self, backend):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = [1]
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.all_special_tokens = []
        tokenizer.decode.return_value = 'answer'
        model = Mock()
        model.generation_config.eos_token_id = 2
        return Qwen(model, tokenizer, backend, ModelConfig(backend=backend))

    def test_torch_greedy_and_sampled_paths(self):
        import torch
        qwen = self.make_model('cpu')
        qwen.model.generate.return_value = torch.tensor([[1, 3, 2]])
        with patch('torch.manual_seed') as seed:
            qwen.generate_with_stats('prompt', temperature=.7, top_p=.9, seed=12)
            seed.assert_called_once_with(12)
        kwargs = qwen.model.generate.call_args.kwargs
        self.assertTrue(kwargs['do_sample'])
        self.assertEqual((kwargs['temperature'], kwargs['top_p']), (.7, .9))
        qwen.generate_with_stats('prompt')
        kwargs = qwen.model.generate.call_args.kwargs
        self.assertFalse(kwargs['do_sample'])
        self.assertIsNone(kwargs['temperature'])
        self.assertIsNone(kwargs['top_p'])

    @unittest.skipUnless(find_spec("mlx_lm"), "requires optional MLX dependencies")
    def test_mlx_sampler_receives_parameters(self):
        qwen = self.make_model('mlx')
        chunk = SimpleNamespace(text='answer', generation_tokens=1, finish_reason='stop')
        with patch('mlx_lm.stream_generate', return_value=iter([chunk])), \
             patch('mlx_lm.sample_utils.make_sampler') as sampler, \
             patch('mlx.core.random.seed') as seed:
            result = qwen.generate('prompt', temperature=.8, top_p=.92, seed=33)
            sampler.assert_called_once_with(temp=.8, top_p=.92, top_k=0)
            seed.assert_called_once_with(33)
            self.assertEqual(result, 'answer')

    def test_presence_penalty_ignores_prompt_and_counts_presence_once(self):
        import torch
        qwen = self.make_model('cpu')
        qwen.model.generate.return_value = torch.tensor([[1, 3, 2]])
        qwen.generate_with_stats('prompt', temperature=1, top_k=20, presence_penalty=1.5)
        kwargs = qwen.model.generate.call_args.kwargs
        self.assertEqual(kwargs['top_k'], 20)
        scores = kwargs['logits_processor'][0](torch.tensor([[1, 3, 3]]), torch.zeros((1, 5)))
        self.assertEqual(scores.tolist(), [[0., 0., 0., -1.5, 0.]])

    @unittest.skipUnless(find_spec("mlx_lm"), "requires optional MLX dependencies")
    def test_mlx_presence_penalty_ignores_last_prompt_token(self):
        import mlx.core as mx
        qwen = self.make_model('mlx')
        chunk = SimpleNamespace(text='answer', generation_tokens=1, finish_reason='stop')
        with patch('mlx_lm.stream_generate', return_value=iter([chunk])) as generate:
            qwen.generate_with_stats('prompt', temperature=1, top_k=20, presence_penalty=1.5)
            processor = generate.call_args.kwargs['logits_processors'][0]
            scores = processor(mx.array([1, 3, 3]), mx.zeros((1, 5)))
            self.assertEqual(scores.tolist(), [[0., 0., 0., -1.5, 0.]])

    def test_answer_prefill_is_input_not_counted_as_generated_tokens(self):
        import torch
        from qwen import ANSWER_PREFIX
        qwen = self.make_model('cpu')
        qwen.config = ModelConfig(backend='cpu', answer_prefix=True)
        qwen.tokenizer.encode.return_value = [7, 8]
        qwen.tokenizer.decode.return_value = '[1, 2]]</answer>'
        qwen.model.generate.return_value = torch.tensor([[1, 7, 8, 3, 2]])
        result = qwen.generate_with_stats('prompt')
        self.assertEqual(result.text, ANSWER_PREFIX + '[1, 2]]</answer>')
        self.assertEqual((result.prompt_tokens, result.generated_tokens), (3, 2))
        self.assertEqual(qwen.model.generate.call_args.kwargs['input_ids'].tolist(), [[1, 7, 8]])
        with self.assertRaises(ValueError):
            qwen.generate_with_stats('prompt', enable_thinking=True)
        with self.assertRaises(ValueError):
            ModelConfig(answer_prefix=True, thinking=True)

    def test_invalid_sampling_parameters(self):
        qwen = self.make_model('cpu')
        for kwargs in ({'temperature': -1}, {'temperature': float('nan')},
                       {'temperature': True}, {'top_p': 0}, {'top_p': 1.1}, {'seed': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                qwen.generate_with_stats('prompt', **kwargs)


if __name__ == '__main__':
    unittest.main()
