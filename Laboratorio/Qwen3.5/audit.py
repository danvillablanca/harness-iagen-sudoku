"""Offline audit of datasets, teacher exports and baseline evidence (no weights)."""
import argparse
from collections import Counter
import json
from pathlib import Path

from baseline import summarize
from dataset import load
from experiment import digest, load_samples, write_json
from reward import reward


def audit_dataset(directory):
    splits = {s: load(Path(directory) / f'{s}.jsonl') for s in ('train', 'eval', 'test')}
    ids, prompts, puzzles = set(), set(), set()
    groups = {}
    for split, records in splits.items():
        groups[split] = dict(Counter(f"{r['size']}x{r['size']}/{r['level']}" for r in records))
        for task in records:
            key = digest([task['size'], task['puzzle']])
            if task['split'] != split or task['id'] in ids or task['prompt'] in prompts or key in puzzles:
                raise ValueError(f"Split mismatch or duplicate task: {task['id']}")
            ids.add(task['id']); prompts.add(task['prompt']); puzzles.add(key)
            answer = '<answer>' + json.dumps(task['solution']) + '</answer>'
            if reward(task['prompt'], answer) != 1:
                raise ValueError(f"Invalid reference: {task['id']}")
    return splits, {'tasks': {s: len(r) for s, r in splits.items()}, 'by_level': groups,
                    'exact_overlap': 0, 'references_verified': len(ids),
                    'limitation': 'Exact duplicate check; Sudoku symmetry classes are not excluded.'}


def audit_traces(directory, train):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    tasks = {r['id']: r for r in train}
    selected = {r['id']: r for r in manifest['tasks']}
    candidates = load_samples(directory / 'candidates.jsonl')
    raw = {}
    for row in candidates:
        task = tasks[row['task_id']]
        if row['task_id'] not in selected or row['prompt'] != task['prompt']:
            raise ValueError('Teacher candidate is not from the selected training set.')
        if row['candidate_id'] in raw or row['run_id'] != manifest['run_id']:
            raise ValueError('Duplicate candidate or mismatched teacher run.')
        if reward(task['prompt'], row['completion']) != row['reward']:
            raise ValueError('Teacher reward failed re-verification.')
        raw[row['candidate_id']] = row
    sft, dpo = load_samples(directory / 'sft.jsonl'), load_samples(directory / 'dpo.jsonl')
    from traces import export_task
    expected_sft, expected_dpo = [], []
    for task in manifest['tasks']:
        rows = sorted([r for r in candidates if r['task_id'] == task['id']], key=lambda r: r['attempt'])
        a, b = export_task(task, rows)
        expected_sft.extend(a); expected_dpo.extend(b)
    if sft != expected_sft or dpo != expected_dpo:
        raise ValueError('Derived exports differ from verified candidate log.')
    for row in sft:
        if reward(row['prompt'], row['completion']) != 1:
            raise ValueError('An SFT response failed verification.')
    for row in dpo:
        if reward(row['prompt'], row['chosen']) != 1 or reward(row['prompt'], row['rejected']) != 0:
            raise ValueError('A DPO pair failed verification.')
        if raw[row['chosen_id']]['task_id'] != raw[row['rejected_id']]['task_id']:
            raise ValueError('DPO responses come from different tasks.')
    return {'model': manifest['model_config']['model_id'], 'samples': len(candidates),
            'samples_planned': len(selected) * manifest['k'],
            'complete': len(candidates) == len(selected) * manifest['k'],
            'sft': len(sft), 'dpo': len(dpo), 'sft_tasks': len({r['task_id'] for r in sft}),
            'dpo_tasks': len({r['task_id'] for r in dpo}), 'reverified': True}


def audit_baselines(files, held_out):
    tasks = {r['id']: r for r in held_out}
    samples, manifests, task_sets = [], [], []
    for path in files:
        path = Path(path)
        manifest = json.loads(path.with_suffix('.manifest.json').read_text())
        selected = {r['id']: r for r in manifest['tasks']}
        task_sets.append({(task_id, task['prompt']) for task_id, task in selected.items()})
        for task in selected.values():
            if task['id'] not in tasks or task['prompt'] != tasks[task['id']]['prompt']:
                raise ValueError('Baseline manifest does not match the held-out dataset.')
        rows = load_samples(path)
        for row in rows:
            task = selected[row['task_id']]
            if row['run_id'] != manifest['run_id'] or row['prompt_sha256'] != digest(task['prompt']):
                raise ValueError('Baseline sample has incompatible provenance.')
            if row['reward'] != reward(task['prompt'], row['completion']):
                raise ValueError('Baseline reward failed re-verification.')
            for key in ('split', 'size', 'level', 'empty_cells'):
                if row[key] != task[key]:
                    raise ValueError('Baseline metadata disagrees with task.')
        samples.extend(rows); manifests.append(manifest)
    paired = all(s == task_sets[0] for s in task_sets) if task_sets else False
    if not paired:
        raise ValueError('Models were evaluated on different task sets.')
    n = min(m['settings']['n'] for m in manifests)
    rows = summarize(samples, [k for k in (1, 2, 4, 8) if k <= n], planned=manifests)
    return {'paired_tasks': True, 'models': sorted({m['settings']['model'] for m in manifests}),
            'complete': all(not r['provisional'] for r in rows), 'metrics': rows}


def audit(dataset, traces, baselines):
    splits, data_check = audit_dataset(dataset)
    trace_check = audit_traces(traces, splits['train'])
    base_check = audit_baselines(baselines, splits['eval'] + splits['test'])
    gaps = []
    if not trace_check['complete']:
        gaps.append('Teacher sampling incomplete')
    if not trace_check['sft']:
        gaps.append('No verified SFT responses')
    if not trace_check['dpo']:
        gaps.append('No verified DPO pairs')
    if not base_check['complete']:
        gaps.append('Baseline sampling incomplete')
    if len(base_check['models']) < 2:
        gaps.append('Need at least two models including the small target')
    return {'dataset': data_check, 'traces': trace_check, 'baseline': base_check,
            'artifact_gaps': gaps, 'artifacts_complete_for_selected_scope': not gaps,
            'scope_note': 'Completion is relative to the selected tasks. A small pilot does not establish final calibration.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/sudoku'))
    parser.add_argument('--traces', type=Path, required=True)
    parser.add_argument('--baselines', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset, args.traces, args.baselines)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))
    return 0 if result['artifacts_complete_for_selected_scope'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
