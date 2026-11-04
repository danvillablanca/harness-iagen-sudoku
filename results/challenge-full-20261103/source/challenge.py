"""Run challenge steps 3–4 sequentially on the installed 9B teacher and 2B target.

One model process at a time. Pilot selects one task per size/level and four
samples; full uses all train/eval tasks and eight samples. Neither trains weights.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from audit import audit, audit_dataset
from baseline import output_path
from experiment import prepare_manifest, write_json

MODELS = {'2b': 'Qwen/Qwen3.5-2B', '9b': 'mlx-community/Qwen3.5-9B-4bit'}


def plan(args):
    root = Path(__file__).parent
    samples = args.samples or (4 if args.profile == 'pilot' else 8)
    per_level = args.per_level if args.per_level is not None else (1 if args.profile == 'pilot' else None)
    selection = ['--per-level', str(per_level)] if per_level is not None else []
    if args.sizes:
        selection += ['--sizes', *map(str, args.sizes)]
    common = ['--backend', args.backend, '--no-thinking', '--answer-prefix', '--temperature', str(args.temperature),
              '--top-p', str(args.top_p), '--top-k', '20', '--presence-penalty', '1.5',
              '--seed', str(args.seed), '--max-output-tokens', str(args.max_output_tokens)]
    teacher = args.out / 'traces-9b'
    stages = [('teacher', [sys.executable, str(root / 'traces.py'),
                          '--input', str(args.dataset / 'train.jsonl'), '--out', str(teacher),
                          '--model', MODELS['9b'], '--k', str(samples), *selection, *common],
               teacher / 'manifest.json', teacher / 'summary.json')]
    baseline_files = []
    for label in ('9b', '2b'):
        path = output_path(args.out / 'baseline', MODELS[label], False)
        baseline_files.append(path)
        stages.append((f'baseline-{label}', [sys.executable, str(root / 'baseline.py'), 'run',
                       '--tasks', str(args.dataset / f'{args.split}.jsonl'), '--out', str(path.parent),
                       '--model-id', MODELS[label], '--n', str(samples), *selection, *common],
                       path.with_suffix('.manifest.json'), path.with_suffix('.summary.json')))
    return stages, baseline_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/sudoku-final'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', choices=['pilot', 'full'], default='pilot')
    parser.add_argument('--split', choices=['eval', 'test'], default='eval')
    parser.add_argument('--per-level', type=int)
    parser.add_argument('--sizes', type=int, nargs='+', choices=[4, 6, 9])
    parser.add_argument('--samples', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-output-tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=.7)
    parser.add_argument('--top-p', type=float, default=.8)
    parser.add_argument('--backend', choices=['auto', 'mlx', 'cuda', 'mps', 'cpu'], default='mlx')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.samples is not None and args.samples < 2:
        parser.error('samples must be >= 2')
    if args.per_level is not None and args.per_level < 1:
        parser.error('per-level must be positive')
    _, dataset_audit = audit_dataset(args.dataset)
    stages, baseline_files = plan(args)
    specification = {'schema': 1, 'profile': args.profile, 'dataset': str(args.dataset.resolve()),
                     'commands': [cmd for _, cmd, _, _ in stages], 'dataset_audit': dataset_audit}
    if args.dry_run:
        print(json.dumps(specification, indent=2))
        return 0
    prepare_manifest(args.out / 'pipeline.json', specification, resume=args.resume)
    state_path = args.out / 'pipeline-status.json'
    try:
        for name, command, manifest, summary in stages:
            write_json(state_path, {'status': 'running', 'stage': name})
            if args.resume and manifest.exists():
                command = [*command, '--resume']
            print(f'\n=== {name} ===', flush=True)
            subprocess.run(command, check=True)
        result = audit(args.dataset, args.out / 'traces-9b', baseline_files)
        write_json(args.out / 'audit.json', result)
        write_json(args.out / 'metrics.json', result['baseline']['metrics'])
        from baseline import format_table
        ks = [k for k in (1, 2, 4, 8) if f'pass@{k}' in result['baseline']['metrics'][0]]
        (args.out / 'metrics.md').write_text(format_table(result['baseline']['metrics'], ks) + '\n')
        status = 'complete' if result['artifacts_complete_for_selected_scope'] else 'needs_more_data'
        write_json(state_path, {'status': status, 'artifact_gaps': result['artifact_gaps']})
        print(json.dumps({'status': status, 'traces': result['traces'],
                          'artifact_gaps': result['artifact_gaps']}, indent=2))
        return 0 if status == 'complete' else 1
    except BaseException:
        write_json(state_path, {'status': 'incomplete', 'stage': name})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
