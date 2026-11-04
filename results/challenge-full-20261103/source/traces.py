"""Sample the larger teacher on train only; verify and export resumable SFT/DPO."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from config import ModelConfig
from dataset import load
from evaluation import evaluate_response
from experiment import (digest, load_samples, prepare_manifest, provenance, sample_seed,
                        write_json, write_jsonl)
from reward import reward
from sudoku import task_from_prompt


def load_train(path):
    records, ids, prompts = [], set(), set()
    for line_number, record in enumerate(load(path), 1):
        if record.get('split') != 'train':
            raise ValueError(f'Line {line_number}: only explicit split=train records are allowed.')
        task_id, prompt = record.get('id'), record.get('prompt')
        if not isinstance(task_id, str) or not task_id.strip() or not isinstance(prompt, str):
            raise ValueError(f'Line {line_number}: id and prompt must be text.')
        if task_id in ids or prompt in prompts:
            raise ValueError(f'Line {line_number}: duplicate task id or prompt.')
        reward(prompt, '')
        ids.add(task_id)
        prompts.add(prompt)
        # The model never receives a reference solution or private puzzle fields.
        records.append({'id': task_id, 'prompt': prompt, 'size': record.get('size'),
                        'level': record.get('level')})
    if not records:
        raise ValueError('The input dataset is empty.')
    return records


def training_text(text):
    # Qwen's generation prompt supplies the opening token. Restore it only in
    # the training export; candidates retain the exact generated continuation.
    if '</think>' in text and '<think>' not in text:
        return '<think>\n' + text
    return text


def export_task(task, candidates):
    unique = {}
    for row in candidates:
        unique.setdefault(row['completion'], row)
    accepted = [row for row in unique.values() if row['reward'] == 1.0]
    rejected = [row for row in unique.values() if row['reward'] == 0.0]
    sft = []
    for row in accepted:
        completion = training_text(row['completion'])
        sft.append({'task_id': task['id'], 'prompt': task['prompt'], 'completion': completion,
                    'candidate_id': row['candidate_id'],
                    'messages': [{'role': 'user', 'content': task['prompt']},
                                 {'role': 'assistant', 'content': completion}]})
    dpo = [{'task_id': task['id'], 'prompt': task['prompt'],
            'chosen': training_text(accepted[0]['completion']),
            'rejected': training_text(row['completion']),
            'chosen_id': accepted[0]['candidate_id'], 'rejected_id': row['candidate_id']}
           for row in rejected] if accepted else []
    return sft, dpo


def rebuild_exports(records, candidates, output, k, *, status='running'):
    """Rebuild derived files from the durable sample log, including partial tasks."""
    grouped = {task['id']: [] for task in records}
    for row in candidates:
        grouped[row['task_id']].append(row)
    sft, dpo = [], []
    summary = {'status': status, 'tasks_planned': len(records), 'tasks_completed': 0,
               'samples_planned': len(records) * k, 'samples': len(candidates),
               'accepted_samples': sum(row['reward'] == 1 for row in candidates),
               'token_limit_reached': sum(row['finish_reason'] == 'length' for row in candidates),
               'tasks_without_success': [], 'tasks_without_rejection': [], 'by_level': {}}
    for task in records:
        rows = sorted(grouped[task['id']], key=lambda row: row['attempt'])
        accepted, pairs = export_task(task, rows)
        sft.extend(accepted)
        dpo.extend(pairs)
        complete = len(rows) == k
        summary['tasks_completed'] += complete
        if complete and not accepted:
            summary['tasks_without_success'].append(task['id'])
        if complete and all(row['reward'] == 1 for row in rows):
            summary['tasks_without_rejection'].append(task['id'])
        key = f"{task.get('size')}x{task.get('size')}/{task.get('level')}"
        group = summary['by_level'].setdefault(key, {'tasks': 0, 'completed': 0, 'samples': 0,
                                                      'accepted': 0, 'sft': 0, 'dpo': 0})
        group['tasks'] += 1
        group['completed'] += complete
        group['samples'] += len(rows)
        group['accepted'] += sum(row['reward'] == 1 for row in rows)
        group['sft'] += len(accepted)
        group['dpo'] += len(pairs)
    summary.update(sft_records=len(sft), dpo_pairs=len(dpo))
    summary['tasks_with_sft'] = len({row['task_id'] for row in sft})
    summary['tasks_with_dpo'] = len({row['task_id'] for row in dpo})
    write_jsonl(output / 'sft.jsonl', sft)
    write_jsonl(output / 'dpo.jsonl', dpo)
    write_json(output / 'summary.json', summary)
    return summary


def generate(records, model, output, *, k, temperature, top_p, seed, model_id, max_tokens,
             top_k=20, presence_penalty=1.5, resume=False, run_id=None, export_only=False):
    output = Path(output)
    path = output / 'candidates.jsonl'
    if path.exists() and not resume:
        raise ValueError('Candidates already exist; use resume.')
    candidates = load_samples(path, repair_tail=resume)
    tasks = {task['id']: task for task in records}
    done = set()
    for row in candidates:
        key = (row['task_id'], row['attempt'])
        if key in done or key[0] not in tasks or not 0 <= key[1] < k:
            raise ValueError('Duplicate or foreign candidate in resume log.')
        expected = {'model_id': model_id, 'temperature': temperature, 'top_p': top_p,
                    'top_k': top_k, 'presence_penalty': presence_penalty, 'run_id': run_id,
                    'seed': sample_seed(seed, *key), 'prompt': tasks[key[0]]['prompt'],
                    'candidate_id': f'{key[0]}:{key[1]}'}
        if any(row.get(name) != value for name, value in expected.items()):
            raise ValueError('Candidate does not match this experiment.')
        if row['reward'] != reward(row['prompt'], row['completion']):
            raise ValueError('Saved candidate reward disagrees with the verifier.')
        done.add(key)
    if export_only:
        return rebuild_exports(records, candidates, output, k,
                               status='complete' if len(done) == len(records) * k else 'incomplete')
    rebuild_exports(records, candidates, output, k)
    try:
        with path.open('a', encoding='utf-8') as raw:
            for task in records:
                public = task_from_prompt(task['prompt'])
                for attempt in range(k):
                    if (task['id'], attempt) in done:
                        continue
                    actual_seed = sample_seed(seed, task['id'], attempt)
                    start = time.perf_counter()
                    result = model.generate_with_stats(
                        task['prompt'], max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=actual_seed,
                        top_k=top_k, presence_penalty=presence_penalty,
                        progress_callback=lambda tokens, tps: print(f'  {tokens} tokens ({tps:.1f}/s)', flush=True))
                    report = evaluate_response(result.text, public)
                    row = {'candidate_id': f"{task['id']}:{attempt}", 'task_id': task['id'],
                           'prompt': task['prompt'], 'completion': result.text,
                           'reward': reward(task['prompt'], result.text), 'run_id': run_id,
                           'size': task.get('size'), 'level': task.get('level'),
                           'model_id': model_id, 'temperature': temperature, 'top_p': top_p,
                           'top_k': top_k, 'presence_penalty': presence_penalty,
                           'seed': actual_seed, 'attempt': attempt,
                           'generated_tokens': result.generated_tokens, 'prompt_tokens': result.prompt_tokens,
                           'finish_reason': result.finish_reason, 'format_ok': report['parsed'].format_ok,
                           'parser_issues': report['parsed'].issues, 'checks': report['checks'],
                           'generation_seconds': time.perf_counter() - start}
                    raw.write(json.dumps(row, ensure_ascii=False) + '\n')
                    raw.flush()
                    candidates.append(row)
                    rebuild_exports(records, candidates, output, k)
                    print(f"{task['id']} [{attempt + 1}/{k}] reward={row['reward']} "
                          f"tokens={result.generated_tokens} ({row['generation_seconds']:.1f}s)", flush=True)
    except BaseException:
        rebuild_exports(records, candidates, output, k, status='incomplete')
        raise
    return rebuild_exports(records, candidates, output, k, status='complete')


def main():
    from baseline import select_tasks
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/sudoku/train.jsonl'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default='mlx-community/Qwen3.5-9B-4bit')
    parser.add_argument('--backend', choices=['mlx', 'cuda', 'mps', 'cpu', 'auto'], default='mlx')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--presence-penalty', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-output-tokens', type=int, default=2048)
    parser.add_argument('--max-prompt-tokens', type=int, default=2048)
    parser.add_argument('--thinking', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--per-level', type=int, help='balanced pilot: N tasks per size and level')
    parser.add_argument('--sizes', type=int, nargs='+', choices=[4, 6, 9])
    parser.add_argument('--max-tasks', type=int, help='first N selected training records')
    parser.add_argument('--answer-prefix', action='store_true', help='prefill <answer> and [; no solution values')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export-only', action='store_true', help='with --resume: rebuild exports, no model loading')
    parser.add_argument('--dry-run', action='store_true', help='validate and preview; creates no files')
    args = parser.parse_args()
    if args.k < 2 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('k must be >= 2 and temperature finite and > 0')
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error('top-p must be in (0, 1]')
    if args.top_k < 0 or not 0 <= args.presence_penalty <= 2:
        parser.error('top-k must be >= 0 and presence-penalty in [0, 2]')
    if any(value is not None and value < 1 for value in (args.max_tasks, args.per_level)):
        parser.error('task limits must be positive')
    if args.export_only and not args.resume:
        parser.error('--export-only requires --resume')
    records = select_tasks(load_train(args.input), args.per_level, args.sizes)
    if args.max_tasks:
        records = records[:args.max_tasks]
    if not records:
        parser.error('No tasks selected')
    config = ModelConfig(model_id=args.model, backend=args.backend, thinking=args.thinking,
                         max_output_tokens=args.max_output_tokens, max_prompt_tokens=args.max_prompt_tokens, answer_prefix=args.answer_prefix)
    specification = {'schema': 2, 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                     'tasks': records, 'model_config': asdict(config), 'k': args.k,
                     'temperature': args.temperature, 'top_p': args.top_p, 'seed': args.seed,
                     'top_k': args.top_k, 'presence_penalty': args.presence_penalty,
                     'filter': 'reward(prompt, raw completion) == 1; final board and format only',
                     'pairing': 'first unique accepted with every unique rejected of same task',
                     'export_transform': 'restore opening <think> when generated continuation contains only closing marker',
                     'provenance': provenance(args.model, 'traces.py')}
    print(f'{len(records)} training tasks, {args.k} samples each -> {args.out}', flush=True)
    if args.dry_run:
        return 0
    if args.out.exists() and not args.resume:
        parser.error('Output directory exists; choose a new one or --resume')
    manifest = prepare_manifest(args.out / 'manifest.json', specification, resume=args.resume)
    candidates = load_samples(args.out / 'candidates.jsonl', repair_tail=args.resume)
    model = None
    try:
        if len(candidates) < len(records) * args.k and not args.export_only:
            from models import load_model
            write_json(args.out / 'summary.json', {'status': 'loading_model'})
            model = load_model(config)
        return_value = generate(records, model, args.out, k=args.k, temperature=args.temperature,
                                top_p=args.top_p, seed=args.seed, model_id=args.model,
                                max_tokens=args.max_output_tokens, top_k=args.top_k,
                                presence_penalty=args.presence_penalty, resume=args.resume,
                                run_id=manifest['run_id'], export_only=args.export_only)
    except BaseException:
        state_path = args.out / 'summary.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        write_json(state_path, {**state, 'status': 'incomplete'})
        raise
    print(json.dumps(return_value, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
