"""Launch or resume this frozen full run; never start a second active worker."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    snapshot = json.loads((root / 'snapshot.json').read_text())
    for name, expected in snapshot['source_hashes_frozen'].items():
        if hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Frozen source changed: {name}')
    manifest = json.loads((root / 'input/manifest.json').read_text())
    for split, info in manifest['files'].items():
        if hashlib.sha256((root / 'input' / f'{split}.jsonl').read_bytes()).hexdigest() != info['sha256']:
            raise SystemExit(f'Frozen input changed: {split}')
    python = snapshot['python']
    command = [python, '-u', str(root / 'source/challenge.py'), '--dataset', str(root / 'input'),
               '--out', str(root), '--profile', 'full', '--samples', '8', '--seed', '42',
               '--max-output-tokens', '1024', '--temperature', '0.7', '--top-p', '0.8',
               '--backend', 'mlx', '--split', 'eval']
    if args.resume:
        command.append('--resume')
    env = {**os.environ, **snapshot['environment']}
    if args.dry_run:
        return subprocess.call([*command, '--dry-run'], cwd=snapshot['source_checkout'], env=env)
    with (root / 'control.lock').open('a') as control_lock, (root / 'launch.lock').open('a') as lock:
        fcntl.flock(control_lock, fcntl.LOCK_EX)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control_path = root / 'control.json'
        if control_path.exists() and json.loads(control_path.read_text()).get('desired_state') == 'paused':
            raise SystemExit('Run intentionally paused. Use R in the dashboard or run_control.py resume.')
        launch_path = root / 'launch.json'
        previous = json.loads(launch_path.read_text()) if launch_path.exists() else None
        if previous:
            try:
                existing = psutil.Process(previous['pid'])
                if existing.is_running() and existing.status() != psutil.STATUS_ZOMBIE and abs(
                        existing.create_time() - previous['process_created']) < 1:
                    raise SystemExit(f"Run is already active (PID {existing.pid}).")
            except psutil.NoSuchProcess:
                pass
            if not args.resume:
                raise SystemExit('Use --resume for an existing run.')
        elif args.resume:
            raise SystemExit('No earlier launch to resume.')
        scripts = {str(root / 'source' / name)
                   for name in ('challenge.py', 'traces.py', 'baseline.py')}
        for proc in psutil.process_iter(['cmdline', 'status']):
            try:
                if (proc.info['status'] != psutil.STATUS_ZOMBIE
                        and scripts.intersection(proc.info['cmdline'] or [])):
                    raise SystemExit(f'Inference for this run is still active (PID {proc.pid}).')
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        with (root / 'run.log').open('a', buffering=1) as log:
            log.write('\nLaunch UTC: ' + datetime.now(timezone.utc).isoformat() + '\n')
            log.flush()
            process = subprocess.Popen(['/usr/bin/caffeinate', '-is', *command],
                                       cwd=snapshot['source_checkout'], env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        info = {'pid': process.pid, 'process_created': psutil.Process(process.pid).create_time(),
                'launched_utc': datetime.now(timezone.utc).isoformat(), 'resume': args.resume,
                'command': command, 'environment': snapshot['environment'],
                'log': str(root / 'run.log'), 'attempt': 1 if previous is None else previous['attempt']+1}
        launch_path.write_text(json.dumps(info, indent=2) + '\n')
        print(json.dumps(info, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
