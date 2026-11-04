"""Pause a frozen run at a saved response; resume without changing its protocol.

This controller stays outside source/: existing inference processes need no code
changes. Pause waits for the next checkpoint, then interrupts the isolated job.
An unsaved attempt may have just started; --resume regenerates that same index.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

import psutil


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def write_state(root, state):
    state = {**state, 'updated_utc': datetime.now(timezone.utc).isoformat()}
    path = root / 'control.json'
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as handle:
        json.dump(state, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return state


@contextmanager
def locked(root):
    with (root / 'control.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def identity(proc):
    return {'pid': proc.pid, 'process_created': proc.create_time()}


def matching_process(info):
    if not info:
        return None
    try:
        proc = psutil.Process(info['pid'])
        if (abs(proc.create_time() - info['process_created']) < .01
                and proc.status() != psutil.STATUS_ZOMBIE):
            return proc
    except psutil.NoSuchProcess:
        pass
    return None


def active_processes(root, launch, known=()):
    """Include surviving children even if the recorded pipeline parent died."""
    found = {}
    for info in [launch, *known]:
        proc = matching_process(info)
        if proc:
            found[proc.pid] = proc
            try:
                for child in proc.children(recursive=True):
                    if child.status() != psutil.STATUS_ZOMBIE:
                        found[child.pid] = child
            except psutil.NoSuchProcess:
                pass
    scripts = {str(root / 'source' / name)
               for name in ('challenge.py', 'traces.py', 'baseline.py')}
    for proc in psutil.process_iter(['cmdline', 'status']):
        try:
            if (proc.info['status'] != psutil.STATUS_ZOMBIE
                    and scripts.intersection(proc.info['cmdline'] or [])):
                found[proc.pid] = proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return list(found.values())


def checkpoint_counts(root):
    paths = [root / 'traces-9b/summary.json', *sorted((root / 'baseline').glob('*.summary.json'))]
    return {str(p.relative_to(root)): read_json(p).get('samples', 0) for p in paths}


def flush_results(root):
    """Make already saved responses/exports durable before reporting PAUSED."""
    for folder in (root / 'traces-9b', root / 'baseline'):
        if folder.exists():
            for path in folder.iterdir():
                if path.suffix in ('.json', '.jsonl'):
                    with path.open('rb') as handle:
                        os.fsync(handle.fileno())


def worker_alive(state):
    return matching_process(state.get('controller')) is not None


def request_pause(root):
    root = Path(root).resolve()
    with locked(root):
        state = read_json(root / 'control.json')
        if state.get('status') == 'resuming' and worker_alive(state):
            raise ValueError('Resume is starting; try pause again in a moment.')
        if state.get('status') in ('pause_requested', 'pausing') and worker_alive(state):
            return state
        if read_json(root / 'pipeline-status.json').get('status') == 'complete':
            return {'status': 'complete', 'message': 'The run is already complete.'}
        launch = read_json(root / 'launch.json')
        if not launch:
            raise ValueError('No launch.json: this is not a launched frozen run.')
        processes = active_processes(root, launch, state.get('processes', []))
        state = {'schema': 1, 'desired_state': 'paused', 'status': 'pause_requested',
                 'request_id': uuid.uuid4().hex, 'launch': launch,
                 'processes': [identity(p) for p in processes],
                 'checkpoint_before': checkpoint_counts(root),
                 'message': 'Waiting for the next saved response. Do not shut down yet.'}
        if not processes:
            flush_results(root)
            return write_state(root, {**state, 'status': 'paused',
                                     'message': 'Paused. Saved progress will be kept.'})
        write_state(root, state)
        python = read_json(root / 'snapshot.json')['python']
        try:
            with (root / 'control.log').open('a') as log:
                worker = subprocess.Popen([python, str(Path(__file__).resolve()),
                                           '--run', str(root), '_pause-worker', state['request_id']],
                                          stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                          start_new_session=True)
            state['controller'] = identity(psutil.Process(worker.pid))
            threading.Thread(target=worker.wait, daemon=True).start()
        except Exception as exc:
            write_state(root, {**state, 'status': 'pause_failed', 'message': str(exc)})
            raise
        return write_state(root, state)


def interrupt_job(root, state):
    processes = active_processes(root, state['launch'], state['processes'])
    known = [identity(p) for p in processes]
    parent = matching_process(state['launch'])
    if parent and os.getpgid(parent.pid) == parent.pid and parent.pid != os.getpgrp():
        os.killpg(parent.pid, signal.SIGINT)
    else:
        # An orphaned child still belongs to this run, with its identity checked.
        for proc in processes:
            try:
                proc.send_signal(signal.SIGINT)
            except psutil.NoSuchProcess:
                pass
    return known


def pause_worker(root, request_id, *, poll=.2, stop_timeout=60):
    root = Path(root).resolve()
    while True:
        with locked(root):
            state = read_json(root / 'control.json')
            if state.get('request_id') != request_id or state.get('desired_state') != 'paused':
                return
            processes = active_processes(root, state['launch'], state.get('processes', []))
            counts = checkpoint_counts(root)
            complete = read_json(root / 'pipeline-status.json').get('status') == 'complete'
            advanced = any(n > state['checkpoint_before'].get(k, 0) for k, n in counts.items())
            if complete or not processes:
                flush_results(root)
                write_state(root, {**state, 'status': 'complete' if complete else 'paused',
                                   'checkpoint_after': counts,
                                   'message': 'Run complete.' if complete else 'Paused. Saved progress will be kept.'})
                return
            if advanced:
                flush_results(root)
                state = write_state(root, {**state, 'status': 'pausing',
                                          'message': 'Response saved; stopping inference.'})
                known = interrupt_job(root, state)
                break
        time.sleep(poll)
    deadline = time.monotonic() + stop_timeout
    while active_processes(root, state['launch'], known):
        if time.monotonic() >= deadline:
            raise RuntimeError('Pause requested, but inference has not exited. No forced kill was sent; inspect processes.')
        time.sleep(poll)
    with locked(root):
        state = read_json(root / 'control.json')
        if state.get('request_id') == request_id:
            flush_results(root)
            write_state(root, {**state, 'status': 'paused',
                               'checkpoint_after': checkpoint_counts(root),
                               'message': 'Paused. Saved progress will be kept; safe to shut down.'})


def resume(root):
    root = Path(root).resolve()
    with locked(root):
        state = read_json(root / 'control.json')
        if state.get('status') == 'resuming' and worker_alive(state):
            return state
        if state.get('status') == 'pausing' and worker_alive(state):
            raise ValueError('Pause is stopping inference; wait until PAUSED before resuming.')
        if read_json(root / 'pipeline-status.json').get('status') == 'complete':
            return {'status': 'complete', 'message': 'The run is already complete.'}
        launch = read_json(root / 'launch.json')
        processes = active_processes(root, launch, state.get('processes', []))
        if processes:
            if not matching_process(launch):
                raise ValueError('Surviving inference child detected; no second worker will be launched.')
            if state.get('status') in ('pausing', 'pause_failed'):
                raise ValueError('Pause needs attention; existing inference must exit before a restart.')
            return write_state(root, {**state, 'desired_state': 'running', 'status': 'running',
                                     'request_id': uuid.uuid4().hex,
                                     'message': 'Running. Pending pause cancelled.'})
        state = write_state(root, {**state, 'desired_state': 'running', 'status': 'resuming',
                                  'request_id': uuid.uuid4().hex,
                                  'controller': identity(psutil.Process()),
                                  'message': 'Restarting from saved responses.'})
    python = read_json(root / 'snapshot.json')['python']
    try:
        result = subprocess.run([python, str(root / 'launch.py'), '--resume'],
                                cwd=root, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip())
        with locked(root):
            return write_state(root, {**state, 'status': 'running',
                                     'launch': read_json(root / 'launch.json'),
                                     'message': 'Running. Saved responses preserved.'})
    except Exception as exc:
        with locked(root):
            write_state(root, {**state, 'desired_state': 'paused', 'status': 'resume_failed',
                               'message': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/challenge-full-20261103'))
    parser.add_argument('action', choices=('pause', 'resume', 'status', '_pause-worker'))
    parser.add_argument('request_id', nargs='?')
    args = parser.parse_args()
    root = args.run.resolve()
    try:
        if args.action == '_pause-worker':
            try:
                pause_worker(root, args.request_id)
            except Exception as exc:
                with locked(root):
                    state = read_json(root / 'control.json')
                    if state.get('request_id') == args.request_id:
                        write_state(root, {**state, 'status': 'pause_failed', 'message': str(exc)})
                raise
            return 0
        result = (request_pause(root) if args.action == 'pause' else resume(root)
                  if args.action == 'resume' else read_json(root / 'control.json'))
        print(json.dumps({k: v for k, v in result.items()
                          if k in ('status', 'desired_state', 'message', 'checkpoint_after')}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, psutil.Error) as exc:
        print(f'Control error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
