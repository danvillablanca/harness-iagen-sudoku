"""Terminal dashboard and pause/resume controls for a frozen challenge run.

    python Laboratorio/Qwen3.5/progress.py --run results/challenge-full-20261103

Keys: p pause, r resume, 1 overview, 2 SFT/DPO by level, 3 pass@k, 4 log; q closes
only the viewer. --once prints one snapshot without opening a full-screen UI.
"""
import argparse
from collections import Counter
import curses
import json
from pathlib import Path
import sys
import time

import psutil

from run_control import request_pause, resume, worker_alive

KS = (1, 2, 4, 8)
MODELS = {'9B': 'mlx-community__Qwen3.5-9B-4bit-nothinking',
          '2B': 'Qwen__Qwen3.5-2B-nothinking'}
LEVEL_ORDER = {'easy': 0, 'medium': 1, 'hard': 2}


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours}h {minutes:02d}m' if hours else f'{minutes}m {seconds:02d}s'


def percentage(value):
    return '--' if value is None else f'{value * 100:.1f}%'


def bar(label, done, total, width):
    ratio = done / total if total else 0
    bar_width = max(8, min(40, width - 51))
    filled = round(min(1., max(0., ratio)) * bar_width)
    return (f'{label:<19} [' + '#' * filled + '-' * (bar_width - filled)
            + f'] {100 * ratio:6.2f}%  {done:,}/{total:,}')


def safe_text(value):
    return ''.join(char if char.isprintable() else ' ' for char in str(value))


class Dashboard:
    def __init__(self, root):
        self.root = root
        self.cache = {}
        self.warnings = []
        self.notice = ''
        self.snapshot = json.loads((root / 'snapshot.json').read_text())
        self.planned = self.snapshot['planned']
        self.groups = Counter()
        with (root / 'input/eval.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                self.groups[row['size'], row['level']] += 1

    def read(self, relative):
        path = self.root / relative
        try:
            data = json.loads(path.read_text())
            if data.get('status') == 'loading_model' and relative in self.cache:
                data = {**self.cache[relative], **data}
            self.cache[relative] = data
        except FileNotFoundError:
            if relative in self.cache:
                self.warnings.append(f'{relative} missing; showing last saved values')
        except (OSError, ValueError):
            self.warnings.append(f'{relative} temporarily unreadable; showing last saved values')
        return self.cache.get(relative, {})

    def collect(self):
        self.warnings = []
        self.pipeline = self.read('pipeline-status.json')
        self.teacher = self.read('traces-9b/summary.json')
        self.baselines = {label: self.read(f'baseline/{name}.summary.json')
                          for label, name in MODELS.items()}
        self.audit = self.read('audit.json')
        self.launch = self.read('launch.json')
        self.control = self.read('control.json')
        self.process_alive = None
        if self.launch:
            try:
                proc = psutil.Process(self.launch['pid'])
                self.process_alive = (proc.status() != psutil.STATUS_ZOMBIE and
                                      abs(proc.create_time() - self.launch['process_created']) < 1)
            except psutil.NoSuchProcess:
                self.process_alive = False
            except (psutil.AccessDenied, KeyError):
                pass
        paths = [self.root / 'run.log', self.root / 'traces-9b/summary.json']
        paths += [self.root / f'baseline/{name}.summary.json' for name in MODELS.values()]
        self.last_activity = max((p.stat().st_mtime for p in paths if p.exists()), default=time.time())
        try:
            with (self.root / 'run.log').open('rb') as handle:
                handle.seek(0, 2)
                handle.seek(max(0, handle.tell() - 12000))
                self.log = handle.read().decode('utf-8', errors='replace').splitlines()[-80:]
        except OSError:
            self.log = []
        self.metric_rows = []
        for label in ('9B', '2B'):
            existing = {(r['size'], r['level']): r for r in self.baselines[label].get('metrics', [])}
            for (size, level), planned in sorted(self.groups.items(),
                                                 key=lambda x: (x[0][0], LEVEL_ORDER[x[0][1]])):
                row = existing.get((size, level), {'tasks': 0, 'samples': 0, 'tasks_planned': planned})
                available = all(row.get(f'pass@{k}') is not None for k in KS)
                final = available and row['tasks'] == planned and not row.get('provisional', True)
                self.metric_rows.append({**row, 'label': label, 'size': size, 'level': level,
                                         'available': available, 'final': final})
        return self

    def state(self):
        status = self.pipeline.get('status', 'waiting')
        stage = {'teacher': '9B teacher / train', 'baseline-9b': '9B evaluation',
                 'baseline-2b': '2B evaluation'}.get(self.pipeline.get('stage'), 'final audit')
        if status == 'complete':
            return 'COMPLETE / audit passed', 'good'
        control_status = self.control.get('status')
        if self.control.get('desired_state') == 'paused':
            if control_status == 'paused':
                if self.process_alive:
                    return 'PAUSE MISMATCH / inference is still active', 'bad'
                return 'PAUSED / saved to disk / R resumes', 'accent'
            if control_status in ('pause_failed', 'resume_failed'):
                return 'CONTROL NEEDS ATTENTION / ' + self.control.get('message', ''), 'bad'
            if control_status in ('pause_requested', 'pausing'):
                if not worker_alive(self.control):
                    return 'PAUSE REQUEST INTERRUPTED / press P to retry', 'bad'
                return ('PAUSE REQUESTED / waiting for saved response' if control_status == 'pause_requested'
                        else 'PAUSING / response saved; stopping inference'), 'accent'
        if control_status == 'resuming' and worker_alive(self.control):
            return 'RESUMING / restoring saved progress', 'accent'
        if self.process_alive is False:
            return f'STOPPED / last recorded stage: {stage}', 'bad'
        if status in ('incomplete', 'needs_more_data'):
            return f'{status.upper()} / {stage}', 'bad'
        return f'{status.upper()} / {stage}', 'accent'

    def elapsed(self):
        from datetime import datetime
        start = datetime.fromisoformat(self.snapshot['created_utc']).timestamp()
        end = time.time()
        if self.pipeline.get('status') == 'complete':
            end = (self.root / 'pipeline-status.json').stat().st_mtime
        return duration(end - start)

    def overview(self, width):
        p, t = self.planned, self.teacher
        teacher_done = t.get('samples', 0)
        eval_done = sum(s.get('samples', 0) for s in self.baselines.values())
        final_groups = sum(r['final'] for r in self.metric_rows)
        available_groups = sum(r['available'] for r in self.metric_rows)
        metric_total = len(self.metric_rows) * len(KS)
        lines = [(bar('All generations', teacher_done + eval_done, p['total_samples'], width), 'accent'),
                 (bar('Teacher candidates', teacher_done, p['teacher_samples'], width), 'normal'),
                 (bar('Evaluation answers', eval_done, 2 * p['eval_samples_per_model'], width), 'normal')]
        for label in ('9B', '2B'):
            lines.append((bar(f'  {label} answers', self.baselines[label].get('samples', 0),
                              p['eval_samples_per_model'], width), 'normal'))
        lines.append((bar('Final metric values', final_groups * len(KS), metric_total, width), 'accent'))
        accepted = t.get('accepted_samples', 0)
        lines += [('', 'normal'),
                  (f"SFT unique: {t.get('sft_records', 0):,}   DPO pairs: {t.get('dpo_pairs', 0):,}   "
                   f"Train tasks complete: {t.get('tasks_completed', 0)}/{p['train_tasks']}", 'good'),
                  (f'Verifier accepted: {accepted:,}/{teacher_done:,} '
                   f"({percentage(accepted / teacher_done if teacher_done else None)})   "
                   f"Rejected: {teacher_done-accepted:,}   Truncated: {t.get('token_limit_reached', 0):,}", 'normal'),
                  ('', 'normal'),
                  ('EVAL METRICS - completed tasks only; partial until every task is done', 'accent'),
                  ('Model   Tasks done     pass@1     pass@2     pass@4     pass@8', 'muted')]
        for label in ('9B', '2B'):
            rows = [r for r in self.metric_rows if r['label'] == label]
            complete = sum(r['tasks'] for r in rows)
            values = []
            for k in KS:
                usable = [r for r in rows if r.get(f'pass@{k}') is not None]
                denominator = sum(r.get(f'tasks@{k}', r['tasks']) for r in usable)
                mean = sum(r[f'pass@{k}'] * r.get(f'tasks@{k}', r['tasks']) for r in usable) / denominator if denominator else None
                values.append(percentage(mean))
            lines.append((f'{label:<7} {str(complete)+"/"+str(p["eval_tasks_per_model"]):<12}'
                          + ''.join(f'{v:>11}' for v in values), 'normal'))
        lines += [(f'Metric values available: {available_groups*len(KS)}/{metric_total}; '
                   f'finalized: {final_groups*len(KS)}/{metric_total} '
                   f'({len(self.metric_rows)} size/level/model groups).', 'muted')]
        audit_status = ('PASSED' if self.audit.get('artifacts_complete_for_selected_scope') else
                        'needs attention' if self.audit else 'pending')
        lines += [(f'Final audit: {audit_status}. Percentages above count attempts, not elapsed time.',
                   'good' if audit_status == 'PASSED' else 'muted')]
        if self.log:
            lines.append(('Latest: ' + safe_text(self.log[-1]), 'muted'))
        return lines

    def teacher_levels(self, width):
        t = self.teacher
        samples_per_task = self.planned['teacher_samples'] // self.planned['train_tasks']
        lines = [('9B TEACHER - train only; SFT/DPO are verified exports', 'accent'),
                 ('Board Level    Candidates    Progress   Tasks      Accepted   SFT   DPO', 'muted')]
        for key, row in t.get('by_level', {}).items():
            board, level = key.split('/')
            planned = row['tasks'] * samples_per_task
            lines.append((f"{board:<5} {level:<8} {str(row['samples'])+'/'+str(planned):<13} "
                          f"{row['samples']/planned:>7.1%}   "
                          f"{str(row['completed'])+'/'+str(row['tasks']):<10}"
                          f"{row['accepted']:>8} {row['sft']:>5} {row['dpo']:>5}", 'normal'))
        if not t.get('by_level'):
            lines.append(('Waiting for the teacher checkpoint.', 'muted'))
        lines += [('', 'normal'),
                  (f"Tasks with SFT: {t.get('tasks_with_sft',0)}; with DPO pairs: {t.get('tasks_with_dpo',0)}", 'good'),
                  (f"Completed tasks with no correct answer: {len(t.get('tasks_without_success',[]))}", 'normal'),
                  ('SFT deduplicates identical correct responses within each task.', 'muted'),
                  ('DPO requires a correct and an incorrect response to the same task.', 'muted')]
        return lines

    def metrics(self, width):
        lines = [('PASS@K BY SIZE AND DIFFICULTY - eval only', 'accent'),
                 ('-- = pending; 0.0% is a computed zero score. Partial values can change.', 'muted'),
                 ('Model Board Level    Tasks     pass@1   pass@2   pass@4   pass@8  State', 'muted')]
        for row in self.metric_rows:
            status = 'FINAL' if row['final'] else 'partial' if row['available'] else 'pending'
            values = ''.join(f"{percentage(row.get(f'pass@{k}')):>9}" for k in KS)
            lines.append((f"{row['label']:<6}{str(row['size'])+'x'+str(row['size']):<6}"
                          f"{row['level']:<9}{str(row['tasks'])+'/'+str(row['tasks_planned']):<8}"
                          + values + f'  {status}', 'good' if row['final'] else 'normal' if row['available'] else 'muted'))
        lines += [('', 'normal'),
                  ('Each task contributes only after all 8 planned attempts finish.', 'muted'),
                  ('Four metrics per group: pass@1, pass@2, pass@4, pass@8.', 'muted')]
        return lines

    def lines(self, view, width):
        if view == 'overview':
            return self.overview(width)
        if view == 'levels':
            return self.teacher_levels(width)
        if view == 'metrics':
            return self.metrics(width)
        return [('RECENT INFERENCE LOG - read only', 'accent')] + [(safe_text(s), 'normal') for s in self.log[-40:]]


def watch(screen, dashboard, view, refresh):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.timeout(200)
    colors = {'normal': 0, 'accent': curses.A_BOLD, 'good': 0, 'bad': curses.A_BOLD, 'muted': curses.A_DIM}
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        for index, (name, color) in enumerate((('accent', curses.COLOR_CYAN), ('good', curses.COLOR_GREEN),
                                                ('bad', curses.COLOR_RED), ('muted', curses.COLOR_WHITE)), 1):
            curses.init_pair(index, color, -1)
            colors[name] = curses.color_pair(index) | (curses.A_DIM if name == 'muted' else curses.A_BOLD)
    next_refresh, offset = 0, 0
    dirty = True
    while True:
        if time.monotonic() >= next_refresh:
            dashboard.collect()
            next_refresh = time.monotonic() + refresh
            dirty = True
        if dirty:
            height, width = screen.getmaxyx()
            screen.erase()
            def put(y, text, kind='normal'):
                if 0 <= y < height:
                    try:
                        screen.addnstr(y, 0, safe_text(text), max(0, width-1), colors[kind])
                    except curses.error:
                        pass
            state, state_color = dashboard.state()
            put(0, f'SUDOKU FULL RUN  |  {state}', state_color)
            put(1, f'Elapsed {dashboard.elapsed()}  |  Last activity {duration(time.time()-dashboard.last_activity)} ago'
                   f'  |  {time.strftime("%H:%M:%S")}', 'muted')
            tabs = [('1', 'overview', 'Overview'), ('2', 'levels', 'SFT / DPO'), ('3', 'metrics', 'Metrics'), ('4', 'log', 'Log')]
            put(2, '   '.join(f'{"[" if key == view else " "}{n} {title}{"]" if key == view else " "}'
                             for n, key, title in tabs), 'accent')
            put(3, dashboard.notice or dashboard.control.get('message', ''), 'muted')
            body = dashboard.lines(view, width)
            usable = max(1, height-5)
            offset = min(offset, max(0, len(body)-usable))
            for i, (text, kind) in enumerate(body[offset:offset+usable], 4):
                put(i, text, kind)
            if dashboard.warnings:
                put(height-1, dashboard.warnings[0], 'bad')
            else:
                footer = 'P pause | R resume | 1-4 views | arrows scroll | Q closes viewer'
                if len(body) > usable:
                    footer = f'Rows {offset+1}-{min(offset+usable,len(body))}/{len(body)} | ' + footer
                put(height-1, footer, 'muted')
            screen.refresh()
            dirty = False
        key = screen.getch()
        if key in (ord('q'), ord('Q'), 27):
            return
        if key in map(ord, '1234'):
            view = ('overview','levels','metrics','log')[key-ord('1')]
            offset = 0
            dirty = True
        elif key in (ord('p'), ord('P'), ord('r'), ord('R')):
            try:
                control = request_pause(dashboard.root) if key in (ord('p'), ord('P')) else resume(dashboard.root)
                dashboard.notice = ''
            except Exception as exc:
                dashboard.notice = f'Control error: {exc}'
            next_refresh = 0
            dirty = True
        elif key in (curses.KEY_DOWN, curses.KEY_NPAGE):
            offset += 1 if key == curses.KEY_DOWN else max(1, screen.getmaxyx()[0]-6)
            dirty = True
        elif key in (curses.KEY_UP, curses.KEY_PPAGE):
            offset = max(0, offset - (1 if key == curses.KEY_UP else max(1, screen.getmaxyx()[0]-6)))
            dirty = True
        elif key in (curses.KEY_RESIZE, ord('f'), ord('F')):
            next_refresh = 0
            dirty = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/challenge-full-20261103'))
    parser.add_argument('--refresh', type=float, default=2)
    parser.add_argument('--view', choices=['overview','levels','metrics','log'], default='overview')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not 0.5 <= args.refresh <= 60:
        parser.error('refresh must be between 0.5 and 60 seconds')
    dashboard = Dashboard(args.run.resolve())
    if args.once or not sys.stdout.isatty():
        dashboard.collect()
        print(dashboard.state()[0])
        print('\n'.join(text for text, _ in dashboard.lines(args.view, 110)))
        return 0
    sys.stdout.write('\033]0;Sudoku full run - live progress\007')
    sys.stdout.flush()
    try:
        curses.wrapper(watch, dashboard, args.view, args.refresh)
    except KeyboardInterrupt:
        pass
    print('Progress viewer closed. The job keeps its current running or paused state.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
