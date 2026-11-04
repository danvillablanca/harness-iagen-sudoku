"""Baseline: pass@1 and pass@k per model, board size and difficulty level.

`run` samples `n` completions per task with one model, scores each with
`reward`, and appends one JSON line per sample. Rerunning with --resume and the same
arguments resumes where it stopped (useful when Colab disconnects).
`summary` aggregates one or more of those files into a table.
Run from the repository root:

    python Laboratorio/Qwen3.5/dataset.py --out data/sudoku
    python Laboratorio/Qwen3.5/baseline.py run --tasks data/sudoku/eval.jsonl \\
        --model-id Qwen/Qwen3.5-2B --no-thinking --per-level 5 --n 8
    python Laboratorio/Qwen3.5/baseline.py summary results/*.jsonl
"""

import argparse
import json
import hashlib
import math
from dataclasses import asdict
from collections import defaultdict
from math import comb
from pathlib import Path

from dataset import load
from reward import reward
from timing import timed
from experiment import (digest, load_samples, prepare_manifest, provenance, sample_seed, write_json)
from evaluation import evaluate_response
from sudoku import task_from_prompt

# Settings recorded with every sample; resuming requires the same values.
RUN_KEYS = ("model", "thinking", "temperature", "top_p", "max_tokens", "n", "seed",
            "top_k", "presence_penalty", "backend", "max_prompt_tokens", "answer_prefix")

# Sampling defaults recommended by Qwen for each mode.
SAMPLING = {True: (1.0, 0.95), False: (0.7, 0.8)}


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021)."""
    if any(type(x) is not int for x in (n, c, k)) or not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Requires 0 <= c <= n and 1 <= k <= n.")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def select_tasks(records, per_level=None, sizes=None):
    """First `per_level` tasks of each (size, level), in file order."""
    taken = defaultdict(int)
    selected = []
    for record in records:
        if sizes and record["size"] not in sizes:
            continue
        key = (record["size"], record["level"])
        if per_level is None or taken[key] < per_level:
            taken[key] += 1
            selected.append(record)
    return selected


def output_path(out_dir, model_id, thinking):
    mode = "thinking" if thinking else "nothinking"
    return Path(out_dir) / f"{model_id.replace('/', '__')}-{mode}.jsonl"


def pending(tasks, done, settings):
    """Validate resume identity and return missing stable sample indices."""
    task_map = {task["id"]: task for task in tasks}
    if len(task_map) != len(tasks):
        raise ValueError("Duplicate task ids.")
    finished = defaultdict(set)
    for sample in done:
        if any(sample.get(key) != settings.get(key) for key in RUN_KEYS):
            raise ValueError("Existing results use different settings; use another output.")
        task_id, index = sample["task_id"], sample["sample"]
        if task_id not in task_map or type(index) is not int or not 0 <= index < settings["n"]:
            raise ValueError("Unknown task or invalid sample index in resume log.")
        if index in finished[task_id]:
            raise ValueError("Duplicate sample in resume log.")
        task = task_map[task_id]
        if "prompt" in task:
            if sample.get("prompt_sha256") != digest(task["prompt"]):
                raise ValueError("Task prompt changed since generation.")
            if sample["reward"] != reward(task["prompt"], sample["completion"]):
                raise ValueError("Saved reward disagrees with the verifier.")
        finished[task_id].add(index)
    return [(task, [i for i in range(settings["n"]) if i not in finished[task["id"]]])
            for task in tasks if len(finished[task["id"]]) < settings["n"]]


def task_seed(seed, task_id, sample):
    return sample_seed(seed, task_id, sample)


def run(args):
    temperature, top_p = SAMPLING[args.thinking]
    temperature = args.temperature if args.temperature is not None else temperature
    top_p = args.top_p if args.top_p is not None else top_p
    settings = {
        "model": args.model_id, "thinking": args.thinking,
        "temperature": temperature, "top_p": top_p,
        "max_tokens": args.max_output_tokens, "n": args.n, "seed": args.seed,
        "top_k": args.top_k, "presence_penalty": args.presence_penalty,
        "backend": args.backend, "max_prompt_tokens": args.max_prompt_tokens,
        "answer_prefix": args.answer_prefix,
    }
    tasks = select_tasks(load(args.tasks), args.per_level, args.sizes)
    if not tasks:
        raise ValueError("No tasks selected.")
    if len({t["id"] for t in tasks}) != len(tasks) or len({t["prompt"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task ids or prompts.")
    splits = {task["split"] for task in tasks}
    if len(splits) != 1 or not splits <= {"eval", "test"}:
        raise ValueError("Baseline requires one held-out split: eval or test, never train.")
    for task in tasks:
        public = task_from_prompt(task["prompt"])
        if public['size'] != task['size']:
            raise ValueError('Task size disagrees with the prompt.')
    from config import ModelConfig
    config = ModelConfig(model_id=args.model_id, backend=args.backend,
                         thinking=args.thinking, max_output_tokens=args.max_output_tokens,
                         max_prompt_tokens=args.max_prompt_tokens, answer_prefix=args.answer_prefix)
    public_tasks = [{k: task[k] for k in ('id', 'prompt', 'size', 'level', 'split', 'empty_cells')}
                    for task in tasks]
    path = output_path(args.out, args.model_id, args.thinking)
    specification = {'schema': 2, 'settings': settings, 'model_config': asdict(config),
                     'tasks': public_tasks, 'split': next(iter(splits)),
                     'input_sha256': hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
                     'provenance': provenance(args.model_id, 'baseline.py')}
    print(f"{len(tasks)} tasks x {args.n} samples -> {path}", flush=True)
    if args.dry_run:
        return 0
    if path.exists() and not args.resume:
        raise ValueError('Output exists; use --resume with identical arguments.')
    manifest = prepare_manifest(path.with_suffix('.manifest.json'), specification, resume=args.resume)
    done = load_samples(path, repair_tail=args.resume)
    if any(s.get('run_id') != manifest['run_id'] for s in done):
        raise ValueError('Results belong to another run.')
    work = pending(tasks, done, settings)
    state_path = path.with_suffix('.summary.json')
    ks = sorted({1, *(k for k in (2, 4, 8) if k <= args.n)})

    def checkpoint(status):
        rows = summarize(done, ks, planned=[manifest])
        write_json(state_path, {'status': status, 'run_id': manifest['run_id'],
                               'samples_planned': len(tasks) * args.n, 'samples': len(done),
                               'metrics': rows})

    checkpoint('running')
    try:
        if work:
            from models import load_model
            model, load_seconds = timed(load_model, config)
            print(f"Model loaded in {load_seconds:.1f} s ({model.backend}).", flush=True)
            with path.open('a', encoding='utf-8') as handle:
                for task, missing in work:
                    for index in missing:
                        # A seed belongs to a sample, not a batch or resume boundary.
                        actual_seed = task_seed(args.seed, task['id'], index)
                        result, seconds = timed(
                            model.generate_with_stats, task['prompt'], temperature=temperature,
                            top_p=top_p, top_k=args.top_k, presence_penalty=args.presence_penalty,
                            seed=actual_seed,
                            progress_callback=lambda tokens, tps: print(f'  {tokens} tokens ({tps:.1f}/s)', flush=True))
                        report = evaluate_response(result.text, task_from_prompt(task['prompt']))
                        sample = {**settings, 'run_id': manifest['run_id'],
                                  'task_id': task['id'], 'split': task['split'],
                                  'size': task['size'], 'level': task['level'],
                                  'empty_cells': task['empty_cells'], 'sample': index,
                                  'sample_seed': actual_seed, 'prompt_sha256': digest(task['prompt']),
                                  'reward': reward(task['prompt'], result.text),
                                  'finish_reason': result.finish_reason,
                                  'generated_tokens': result.generated_tokens,
                                  'prompt_tokens': result.prompt_tokens, 'seconds': seconds,
                                  'format_ok': report['parsed'].format_ok,
                                  'parser_issues': report['parsed'].issues, 'checks': report['checks'],
                                  'completion': result.text}
                        handle.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        handle.flush()
                        done.append(sample)
                        checkpoint('running')
                        print(f"[{len(done)}/{len(tasks)*args.n}] {task['id']} #{index}: "
                              f"reward={sample['reward']} ({seconds:.1f}s)", flush=True)
    except BaseException:
        checkpoint('incomplete')
        raise
    checkpoint('complete')
    print(format_table(summarize(done, ks, planned=[manifest]), ks))
    return 0


def summarize(samples, ks=(1, 2, 4, 8), *, planned=()):
    """Fixed-n estimator, macro-averaged over complete tasks only.

    Report coverage and provisional status explicitly. Keep datasets, seeds,
    decoding settings and model runs separate. Duplicated samples are an error.
    Repeated *texts* from independent draws are legitimate and are not removed.
    """
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError('k must contain positive integers.')
    groups, settings_by_run, seen, metadata = {}, {}, set(), {}

    def run_key(sample):
        return sample.get('run_id') or digest({k: sample.get(k) for k in (*RUN_KEYS, 'split')})

    def register(run_id, settings, size, level, task_id):
        signature = {k: settings.get(k) for k in (*RUN_KEYS, 'split')}
        previous = settings_by_run.setdefault(run_id, signature)
        if previous != signature:
            raise ValueError('Conflicting settings within a run.')
        meta = (size, level)
        if metadata.setdefault((run_id, task_id), meta) != meta:
            raise ValueError("Conflicting task metadata within a run.")
        key = (run_id, size, level)
        group = groups.setdefault(key, {'settings': settings, 'tasks': {}})
        group['tasks'].setdefault(task_id, [])
        return group['tasks'][task_id]

    for manifest in planned:
        for task in manifest['tasks']:
            register(manifest['run_id'], {**manifest['settings'], 'split': task['split']},
                     task['size'], task['level'], task['id'])
    planned_ids = {(m['run_id'], t['id']) for m in planned for t in m['tasks']}
    for sample in samples:
        run_id = run_key(sample)
        if planned and (run_id, sample['task_id']) not in planned_ids:
            raise ValueError('Sample is not in the planned task set.')
        identity = run_id, sample['task_id'], sample['sample']
        if identity in seen:
            raise ValueError('Duplicate sample: do not summarize the same result file twice.')
        if type(sample['sample']) is not int or not 0 <= sample['sample'] < sample['n']:
            raise ValueError('Sample index outside the planned n.')
        if sample['reward'] not in (0., 1.):
            raise ValueError('pass@k requires binary rewards.')
        seen.add(identity)
        register(run_id, sample, sample['size'], sample['level'], sample['task_id']).append(sample)

    rows = []
    for (run_id, size, level), group in groups.items():
        settings = group['settings']
        all_tasks = list(group['tasks'].values())
        complete = [s for s in all_tasks if len(s) == settings['n']]
        flat = [sample for task in all_tasks for sample in task]
        row = {'run_id': run_id, **{k: settings.get(k) for k in RUN_KEYS},
               'split': settings.get('split'), 'size': size, 'level': level,
               'tasks': len(complete), 'tasks_planned': len(all_tasks),
               'samples': len(flat), 'provisional': len(complete) != len(all_tasks)}
        for k in ks:
            values = [pass_at_k(len(s), sum(x['reward'] == 1. for x in s), k)
                      for s in complete if len(s) >= k]
            row[f'pass@{k}'] = sum(values) / len(values) if values else None
            row[f'tasks@{k}'] = len(values)
        row['truncated'] = sum(s['finish_reason'] == 'length' for s in flat)/len(flat) if flat else None
        row['format_failures'] = sum(s.get('format_ok') is False for s in flat)
        row['avg_tokens'] = sum(s['generated_tokens'] for s in flat)/len(flat) if flat else None
        correct = sum(s['reward'] == 1 for task in complete for s in task)
        row['calibration_signal'] = ('no_complete_tasks' if not complete else 'zero_success'
                                     if correct == 0 else 'ceiling' if correct == len(complete)*settings['n']
                                     else 'mixed_success')
        row['calibration_note'] = 'Exploratory; more tasks needed' if len(complete) < 20 else 'Inspect difficulty and error types'
        rows.append(row)
    return rows


LEVEL_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def format_table(rows, ks=(1, 2, 4, 8)):
    """Markdown table, ready for the README or the report."""
    rows = sorted(rows, key=lambda r: (r["model"], r["thinking"], r["size"],
                                       LEVEL_ORDER.get(r["level"], 99)))
    header = ["model", "thinking", "board", "level", "tasks",
              *(f"pass@{k}" for k in ks), "truncated", "avg tokens"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["model"], "yes" if row["thinking"] else "no",
                 f"{row['size']}x{row['size']}", row["level"], f"{row['tasks']}/{row['tasks_planned']}"]
        cells += ["-" if row[f"pass@{k}"] is None else f"{row[f'pass@{k}']:.2f}" for k in ks]
        cells += ["-" if row["truncated"] is None else f"{row['truncated']:.0%}",
                  "-" if row["avg_tokens"] is None else f"{row['avg_tokens']:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summary(args):
    samples = [sample for path in args.files for sample in load_samples(path)]
    if not samples:
        raise SystemExit("No samples found.")
    manifests = [Path(path).with_suffix(".manifest.json") for path in args.files]
    planned = [json.loads(path.read_text()) for path in manifests if path.exists()]
    rows = summarize(samples, args.k, planned=planned)
    print(format_table(rows, args.k))
    if args.out:
        write_json(Path(args.out), rows)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Sudoku baseline: pass@1 and pass@k.")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="sample and score one model")
    run_parser.add_argument("--tasks", required=True, help="JSONL from dataset.py")
    run_parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    run_parser.add_argument("--backend", default="auto",
                            choices=["auto", "mlx", "cuda", "mps", "cpu"])
    run_parser.add_argument("--thinking", action=argparse.BooleanOptionalAction,
                            default=False, help="Qwen reasoning mode (default: off)")
    run_parser.add_argument("--n", type=int, default=8, help="samples per task")
    run_parser.add_argument("--batch-size", type=int, choices=[1], default=1,
                            help="one sample at a time for stable seeds and immediate checkpointing")
    run_parser.add_argument("--max-output-tokens", type=int, default=2048)
    run_parser.add_argument("--temperature", type=float, default=None,
                            help="default: 1.0 with thinking, 0.7 without")
    run_parser.add_argument("--top-p", type=float, default=None,
                            help="default: 0.95 with thinking, 0.8 without")
    run_parser.add_argument("--top-k", type=int, default=20)
    run_parser.add_argument("--presence-penalty", type=float, default=1.5)
    run_parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    run_parser.add_argument("--answer-prefix", action="store_true", help="prefill <answer> and [; no solution values")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--per-level", type=int, default=None,
                            help="tasks per size and level (default: all)")
    run_parser.add_argument("--sizes", type=int, nargs="+", default=None, choices=[4, 6, 9])
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--out", default="results", help="output directory")

    summary_parser = commands.add_parser("summary", help="aggregate result files")
    summary_parser.add_argument("files", nargs="+")
    summary_parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    summary_parser.add_argument("--out", default=None, help="optional JSON output")

    args = parser.parse_args()
    if args.command == "run" and (args.n < 1 or args.batch_size < 1):
        parser.error("--n and --batch-size must be positive.")
    if args.command == "run":
        if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature <= 0):
            parser.error("temperature must be finite and > 0")
        if args.top_p is not None and (not math.isfinite(args.top_p) or not 0 < args.top_p <= 1):
            parser.error("top-p must be in (0, 1]")
        if args.top_k < 0 or not 0 <= args.presence_penalty <= 2:
            parser.error("top-k must be >= 0; presence-penalty in [0, 2]")
        if args.per_level is not None and args.per_level < 1:
            parser.error("per-level must be positive")
    return args


def main() -> int:
    args = parse_args()
    return run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    raise SystemExit(main())
