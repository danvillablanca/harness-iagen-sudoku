"""Structural novelty audit for 4x4 train/eval; test remains unexamined here."""
from collections import Counter, defaultdict
from itertools import permutations, product
import json
from pathlib import Path

root = Path(__file__).resolve().parent
run = Path(json.loads((root/'snapshot.json').read_text())['source_run'])
swaps = list(permutations(range(2)))
orders = [tuple(2*band + row for band in bands for row in inside[band])
          for bands in swaps for inside in product(swaps, repeat=2)]
assert len(set(orders)) == 8


def relabel(values):
    mapping = {}
    result = []
    for value in values:
        if value:
            mapping.setdefault(value, len(mapping)+1)
            result.append(mapping[value])
        else:
            result.append(0)
    return tuple(result)


def canonical(grid):
    forms = []
    for transpose in (False, True):
        board = list(zip(*grid)) if transpose else grid
        for rows in orders:
            for cols in orders:
                forms.append(relabel(board[i][j] for i in rows for j in cols))
    return min(forms)


splits = {split:[r for r in map(json.loads,(run/'input'/f'{split}.jsonl').read_text().splitlines()) if r['size']==4]
          for split in ('train','eval')}
keys = {r['id']:canonical(r['puzzle']) for tasks in splits.values() for r in tasks}
for row in splits['train'][:10]:
    grid = row['puzzle']
    transformed = [[5-x if x else 0 for x in rr] for rr in zip(*grid)]
    assert canonical(transformed) == keys[row['id']]
train_by_key = defaultdict(list)
for r in splits['train']:
    train_by_key[keys[r['id']]].append(r['id'])
sft_ids = {r['task_id'] for r in map(json.loads,(root/'teacher-snapshot/sft.jsonl').read_text().splitlines())}
sft_keys = {keys[task_id] for task_id in sft_ids if task_id in keys}
out = {'scope':'4x4 only; train/eval only; no model evaluations and no dataset changes',
       'transformations':'digit renaming; row swaps within bands; band swaps; column swaps within stacks; stack swaps; transpose',
       'caveat':'Structural equivalence does not prove memorization or performance inflation; it limits claims of structurally unseen evaluation tasks.',
       'by_level':{}, 'examples':[]}
for level in ('easy','medium','hard'):
    tr = [r for r in splits['train'] if r['level']==level]
    ev = [r for r in splits['eval'] if r['level']==level]
    sf = [r for r in tr if r['id'] in sft_ids]
    overlaps = [r for r in ev if keys[r['id']] in train_by_key]
    out['by_level'][level] = {'train_tasks':len(tr),'train_classes':len({keys[r['id']] for r in tr}),
                             'sft_tasks':len(sf),'sft_classes':len({keys[r['id']] for r in sf}),
                             'eval_tasks':len(ev),'eval_classes':len({keys[r['id']] for r in ev}),
                             'eval_tasks_equivalent_to_train':len(overlaps),
                             'eval_tasks_equivalent_to_sft':sum(keys[r['id']] in sft_keys for r in ev)}
    if overlaps:
        row = overlaps[0]
        out['examples'].append({'eval_task':row['id'],'train_task':train_by_key[keys[row['id']]][0],
                                'matching_train_tasks_in_sft':[task_id for task_id in train_by_key[keys[row['id']]] if task_id in sft_ids],
                                'canonical_grid':keys[row['id']]})
(root/'symmetry-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
