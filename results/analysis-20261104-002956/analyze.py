"""Read-only analysis of the captured teacher checkpoint. No model imports."""
from collections import Counter, defaultdict
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import sys

OUT = Path(__file__).resolve().parent
META = json.loads((OUT / 'snapshot.json').read_text())
RUN = Path(META['source_run'])
sys.path.insert(0, str(RUN / 'source'))
from audit import audit_dataset, audit_traces
from baseline import pass_at_k
from experiment import sample_seed
from response_parser import parse_answer
from reward import reward


def load(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def ratio(a, b):
    return a / b if b else None


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    return [max(0, center-half), min(1, center+half)]


splits, data_audit = audit_dataset(RUN / 'input')
trace_audit = audit_traces(OUT / 'teacher-snapshot', splits['train'])
rows = load(OUT / 'teacher-snapshot/candidates.jsonl')
sft = load(OUT / 'teacher-snapshot/sft.jsonl')
dpo = load(OUT / 'teacher-snapshot/dpo.jsonl')
tasks = {t['id']: t for t in splits['train']}
raw = {r['candidate_id']: r for r in rows}
assert len(raw) == len(rows)
assert len({(r['task_id'], r['attempt']) for r in rows}) == len(rows)
assert all(r['seed'] == sample_seed(42, r['task_id'], r['attempt']) for r in rows)
snapshot = json.loads((RUN / 'snapshot.json').read_text())
source_ok = all(hashlib.sha256((RUN / 'source' / p).read_bytes()).hexdigest() == h
                for p, h in snapshot['source_hashes_frozen'].items())
manifest = json.loads((RUN / 'input/manifest.json').read_text())
input_ok = all(hashlib.sha256((RUN / 'input' / f'{s}.jsonl').read_bytes()).hexdigest() == v['sha256']
               for s, v in manifest['files'].items())
assert source_ok and input_ok
accepted_boards = 0
for row in rows:
    if row['reward'] != 1:
        continue
    task = tasks[row['task_id']]
    blocks = re.findall(r'<answer>(.*?)</answer>', row['completion'], re.S)
    assert len(blocks) == 1
    grid = json.loads(blocks[0])
    n = task['size']; wanted = set(range(1, n+1))
    assert len(grid) == n and all(len(rr) == n for rr in grid)
    assert all(type(value) is int for rr in grid for value in rr)
    assert all(set(rr) == wanted for rr in grid)
    assert all({grid[i][j] for i in range(n)} == wanted for j in range(n))
    br, bc = task['box_rows'], task['box_columns']
    assert all({grid[i][j] for i in range(top, top+br) for j in range(left,left+bc)} == wanted
               for top in range(0,n,br) for left in range(0,n,bc))
    assert all(value == 0 or grid[i][j] == value for i, rr in enumerate(task['puzzle']) for j,value in enumerate(rr))
    accepted_boards += 1
save('audit.json', {'dataset': data_audit, 'teacher': trace_audit,
                    'duplicate_attempts': 0, 'all_attempt_seeds_verified': True,
                    'accepted_boards_independently_checked': accepted_boards,
                    'frozen_source_unchanged': source_ok, 'frozen_inputs_unchanged': input_ok,
                    'baseline': 'Not started in this full run; this is not a final pipeline audit.'})

by_task = defaultdict(list)
by_group = defaultdict(list)
for r in rows:
    by_task[r['task_id']].append(r)
    by_group[f"{r['size']}x{r['size']}/{r['level']}"].append(r)
diagnostics = {}
for r in rows:
    t = tasks[r['task_id']]
    parsed = parse_answer(r['completion'], r['size'])
    grid = parsed.grid
    checks = r['checks']
    math_ok = all(checks[k] is True for k in ('All cells filled', 'Allowed values', 'Rows', 'Columns', 'Boxes', 'Original clues'))
    category = ('accepted' if r['reward'] == 1 else 'unparseable' if grid is None
                else 'format_only' if math_ok else 'format_and_math' if not parsed.format_ok
                else 'math_with_valid_format')
    hamming = sum(a != b for rr, ss in zip(grid, t['solution']) for a, b in zip(rr, ss)) if grid else None
    diagnostics[r['candidate_id']] = {
        'category': category, 'copied_input': grid == t['puzzle'],
        'wrong_cells': hamming, 'truncated': r['finish_reason'] == 'length',
        'changed_clues': checks['Original clues'] is False,
        'internally_valid_but_changed_clues': checks['Original clues'] is False and
            all(checks[k] is True for k in ('All cells filled', 'Allowed values', 'Rows', 'Columns', 'Boxes')),
        'grid': grid,
    }


def summarize(group_rows):
    tt = defaultdict(list)
    for r in group_rows:
        tt[r['task_id']].append(r)
    complete = [rs for rs in tt.values() if len(rs) == 8]
    counts = [sum(r['reward'] == 1 for r in rs) for rs in complete]
    ds = [diagnostics[r['candidate_id']] for r in group_rows]
    rejected = [r for r in group_rows if r['reward'] == 0]
    truncated = [r for r in group_rows if r['finish_reason'] == 'length']
    stopped = [r for r in group_rows if r['finish_reason'] != 'length']
    tokens = sum(r['generated_tokens'] for r in group_rows)
    seconds = sum(r['generation_seconds'] for r in group_rows)
    unique_sft = sum(any(r['reward'] == 1 for r in rs) for rs in tt.values())
    near_misses = [d['wrong_cells'] for d in ds if d['category'] == 'math_with_valid_format']
    result = {
        'samples': len(group_rows), 'tasks_started': len(tt), 'tasks_complete': len(complete),
        'accepted': sum(r['reward'] == 1 for r in group_rows), 'sft_tasks': unique_sft,
        'pass_k_on_complete_tasks': {str(k): statistics.mean(pass_at_k(8, c, k) for c in counts) if counts else None for k in (1,2,4,8)},
        'task_success_histogram': dict(sorted(Counter(counts).items())),
        'tasks_zero_success': counts.count(0), 'tasks_all_success': counts.count(8),
        'tasks_mixed': sum(0 < c < 8 for c in counts),
        'pass8_task_wilson95': wilson(sum(c > 0 for c in counts), len(counts)),
        'actual_prefix_coverage': {str(k): sum(any(r['reward'] == 1 and r['attempt'] < k for r in rs) for rs in complete) for k in range(1,9)},
        'truncated': len(truncated), 'accepted_truncated': sum(r['reward'] == 1 for r in truncated),
        'truncation_rate': ratio(len(truncated), len(group_rows)),
        'accepted_among_nontruncated': sum(r['reward'] == 1 for r in stopped),
        'nontruncated_samples': len(stopped),
        'categories': dict(Counter(d['category'] for d in ds)),
        'failed_checks': dict(Counter(k for r in rejected for k,v in r['checks'].items() if v is False)),
        'copied_input': sum(d['copied_input'] for d in ds),
        'internally_valid_but_changed_clues': sum(d['internally_valid_but_changed_clues'] for d in ds),
        'wrong_cells_in_formatted_errors': {'median': statistics.median(near_misses) if near_misses else None,
                                           'one': near_misses.count(1), 'two': near_misses.count(2)},
        'zero_success_tasks_identical_all8': sum(len({r['completion'] for r in rs}) == 1 for rs,c in zip(complete,counts) if c == 0),
        'mean_unique_responses_per_complete_task': statistics.mean(len({r['completion'] for r in rs}) for rs in complete) if complete else None,
        'tokens': tokens, 'generation_seconds_saved_attempts': seconds,
        'truncation_token_share': ratio(sum(r['generated_tokens'] for r in truncated), tokens),
        'truncation_time_share': ratio(sum(r['generation_seconds'] for r in truncated), seconds),
        'seconds_per_unique_sft': ratio(seconds, unique_sft),
        'seconds_by_reward': {str(reward): sum(r['generation_seconds'] for r in group_rows if r['reward'] == reward) for reward in (0,1)},
        'accepted_token_histogram': dict(Counter(r['generated_tokens'] for r in group_rows if r['reward'] == 1)),
    }
    return result


groups = {}
for size in (4,6,9):
    for level in ('easy','medium','hard'):
        key = f'{size}x{size}/{level}'
        rs = by_group[key]
        selected = [t for t in tasks.values() if t['size'] == size and t['level'] == level]
        gs = summarize(rs)
        gs.update(tasks_planned=len(selected),
                  holes_min=min(t['empty_cells'] for t in selected),
                  holes_max=max(t['empty_cells'] for t in selected),
                  deducible_by_singles=sum(t['deducible_by_singles'] for t in selected),
                  sft_records=sum(tasks[r['task_id']]['size']==size and tasks[r['task_id']]['level']==level for r in sft),
                  dpo_pairs=sum(tasks[r['task_id']]['size']==size and tasks[r['task_id']]['level']==level for r in dpo))
        groups[key] = gs

negatives = [raw[p['rejected_id']] for p in dpo]
positives = [raw[p['chosen_id']] for p in dpo]
pair_counts = Counter(p['task_id'] for p in dpo)
preferences = {
    'pairs': len(dpo), 'tasks': len(pair_counts),
    'pairs_per_task_histogram': dict(Counter(pair_counts.values())),
    'negative_categories': dict(Counter(diagnostics[r['candidate_id']]['category'] for r in negatives)),
    'negative_truncated': sum(r['finish_reason']=='length' for r in negatives),
    'negative_copied_input': sum(diagnostics[r['candidate_id']]['copied_input'] for r in negatives),
    'chosen_shorter_tokens': sum(a['generated_tokens'] < b['generated_tokens'] for a,b in zip(positives,negatives)),
    'same_tokens': sum(a['generated_tokens'] == b['generated_tokens'] for a,b in zip(positives,negatives)),
    'positive_mean_tokens': statistics.mean(r['generated_tokens'] for r in positives),
    'negative_mean_tokens': statistics.mean(r['generated_tokens'] for r in negatives),
    'source': 'synthetic verifier preferences; no human preference labels',
}
examples = []
selectors = {
    'sudoku_valid_but_clues_changed': lambda r: diagnostics[r['candidate_id']]['internally_valid_but_changed_clues'] and r['size']==4,
    'copied_input': lambda r: diagnostics[r['candidate_id']]['copied_input'],
    'row_valid_column_invalid': lambda r: r['size']==4 and r['format_ok'] and r['checks']['Rows'] and not r['checks']['Columns'] and r['checks']['Original clues'],
    'truncated': lambda r: r['size']==9 and r['finish_reason']=='length',
    'rare_9x9_medium_success': lambda r: r['size']==9 and r['level']=='medium' and r['reward']==1,
}
for label, predicate in selectors.items():
    matches = [r for r in rows if predicate(r)]
    if matches:
        r = matches[0]
        examples.append({'type':label, 'candidate':r, 'puzzle':tasks[r['task_id']]['puzzle'],
                         'reference':tasks[r['task_id']]['solution'], 'diagnostics':diagnostics[r['candidate_id']]})
save('examples.json', examples)
post_counts = Counter()
extra_tasks = set()
for row in rows:
    matches = list(re.finditer(r'<answer>.*?</answer>', row['completion'], re.S))
    if row['finish_reason'] == 'length':
        post_counts['truncated_total'] += 1
        if '</answer>' in row['completion']:
            post_counts['truncated_with_closed_answer'] += 1
        if matches and row['completion'][matches[0].end():].strip():
            post_counts['truncated_with_text_after_first_answer'] += 1
        if re.search(r'\b(wait|re-evaluate|mistake|incorrect|actually solve)\b',row['completion'],re.I):
            post_counts['truncated_self_correction_language'] += 1
    if row['reward'] == 0 and matches:
        if reward(row['prompt'], matches[0].group()) == 1:
            post_counts['rejected_first_answer_block_correct'] += 1
        if any(reward(row['prompt'], match.group()) == 1 for match in matches):
            post_counts['rejected_with_any_correct_answer_block'] += 1
            if not any(r['reward'] == 1 for r in by_task[row['task_id']]):
                extra_tasks.add(row['task_id'])
post = {'counts': dict(post_counts), 'additional_tasks_with_embedded_correct_block': sorted(extra_tasks),
        'method':'Diagnostic only: score each complete answer block independently. Actual verifier still scores the entire original response; no rejected example was promoted to SFT.'}
save('post-answer-diagnostics.json', post)
aggregate = summarize(rows)
completed_groups = [s for s in groups.values() if s['tasks_complete']==s['tasks_planned']]
result = {'snapshot': META, 'interpretation': 'Teacher on train, before any weight training; not held-out baseline accuracy.',
          'overall':aggregate, 'groups':groups, 'sft_records':len(sft), 'preferences':preferences,
          'macro_across_8_complete_groups': {str(k):statistics.mean(s['pass_k_on_complete_tasks'][str(k)] for s in completed_groups) for k in (1,2,4,8)},
          'uncertainty_note':'Wilson intervals treat task-level success in 8 attempts as a Bernoulli outcome across tasks. They are descriptive, do not account for Sudoku symmetries, and do not establish causal effects.',
          'timing_note':'Sums include completed/saved attempts only; omit model loading, pauses, discarded partial attempts and export overhead.',
          'passk_method':'Mean over tasks with all 8 samples of 1 - C(8-c,k)/C(8,k), following Chen et al. 2021; k=1 is sampling accuracy, not greedy decoding.',
          'passk_reference':'https://arxiv.org/html/2107.03374v2'}
result['post_answer_diagnostics'] = post
result['coverage_bounds_at_fixed_budget'] = {
    'completed_tasks_without_success': aggregate['tasks_zero_success'],
    'planned_train_tasks': 450,
    'maximum_final_tasks_with_sft': 450-aggregate['tasks_zero_success'],
    'maximum_final_task_coverage': (450-aggregate['tasks_zero_success'])/450,
    'reason':'Completed tasks receive no more attempts in the frozen run; at most every remaining task can gain one successful response.'}
if (OUT/'symmetry-audit.json').exists():
    result['structural_audit'] = json.loads((OUT/'symmetry-audit.json').read_text())
save('analysis.json', result)
print(json.dumps(result, ensure_ascii=False, indent=2))
