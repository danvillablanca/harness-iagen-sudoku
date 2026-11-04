"""Real subprocess tests of pause/resume; fake responses, never model weights."""
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import psutil

from run_control import (active_processes, identity, matching_process, read_json,
                         request_pause, resume)
from progress import Dashboard


FAKE_JOB = '''import json, time
from pathlib import Path
r = Path(__file__).resolve().parent.parent
folder = r / 'traces-9b'
folder.mkdir(exist_ok=True)
p = folder / 'candidates.jsonl'
def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)
write(r / 'pipeline-status.json', {'status': 'running', 'stage': 'teacher'})
rows = [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
try:
    for i in range(len(rows), 12):
        time.sleep(.18)
        with p.open('a') as f:
            f.write(json.dumps({'sample': i, 'seed': 100+i, 'answer': i*i}) + '\\n')
            f.flush()
        write(folder / 'summary.json', {'samples': i+1, 'status': 'running'})
    write(r / 'pipeline-status.json', {'status': 'complete'})
except KeyboardInterrupt:
    write(r / 'pipeline-status.json', {'status': 'incomplete', 'stage': 'teacher'})
'''


def wait_until(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.03)
    raise AssertionError('Timed out waiting for test subprocess')


@unittest.skipUnless(shutil.which('caffeinate'), 'Frozen launcher uses macOS caffeinate')
class RunControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        source = self.root / 'source'
        source.mkdir()
        (source / 'challenge.py').write_text(FAKE_JOB)
        data = self.root / 'input'
        data.mkdir()
        (data / 'train.jsonl').write_text('{}\n')
        (data / 'manifest.json').write_text(json.dumps({'files': {'train': {
            'sha256': hashlib.sha256(b'{}\n').hexdigest()}}}))
        snapshot = {'python': sys.executable, 'source_checkout': str(self.root),
                    'environment': {}, 'source_hashes_frozen': {
                        'challenge.py': hashlib.sha256(FAKE_JOB.encode()).hexdigest()}}
        (self.root / 'snapshot.json').write_text(json.dumps(snapshot))
        launcher = Path(__file__).resolve().parents[2] / 'results/challenge-full-20261103/launch.py'
        shutil.copyfile(launcher, self.root / 'launch.py')

    def tearDown(self):
        launch = read_json(self.root / 'launch.json')
        state = read_json(self.root / 'control.json')
        for proc in active_processes(self.root, launch, state.get('processes', [])):
            try:
                proc.send_signal(signal.SIGTERM)
            except psutil.NoSuchProcess:
                pass
        controller = matching_process(state.get('controller'))
        if controller and controller.pid != psutil.Process().pid:
            controller.terminate()
        self.temp.cleanup()

    def launch(self):
        result = subprocess.run([sys.executable, str(self.root / 'launch.py')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        wait_until(lambda: read_json(self.root / 'traces-9b/summary.json').get('samples', 0) >= 1)

    def test_saved_pause_blocks_auto_restart_and_resumes_without_duplicates(self):
        self.launch()
        before = read_json(self.root / 'traces-9b/summary.json')['samples']
        requested = subprocess.run([sys.executable, str(Path(__file__).with_name('run_control.py')),
                                    '--run', str(self.root), 'pause'], capture_output=True, text=True)
        self.assertEqual(requested.returncode, 0, requested.stderr)
        state = json.loads(requested.stdout)
        self.assertEqual(state['status'], 'pause_requested')
        wait_until(lambda: read_json(self.root / 'control.json').get('status') == 'paused')
        state = read_json(self.root / 'control.json')
        self.assertFalse(active_processes(self.root, state['launch'], state['processes']))
        saved = (self.root / 'traces-9b/candidates.jsonl').read_bytes()
        self.assertGreater(len(saved.splitlines()), before)
        attempted = subprocess.run([sys.executable, str(self.root / 'launch.py'), '--resume'],
                                   capture_output=True, text=True)
        self.assertNotEqual(attempted.returncode, 0)
        self.assertIn('intentionally paused', attempted.stderr)
        self.assertEqual((self.root / 'traces-9b/candidates.jsonl').read_bytes(), saved)
        self.assertEqual(resume(self.root)['status'], 'running')
        wait_until(lambda: read_json(self.root / 'pipeline-status.json').get('status') == 'complete')
        raw = (self.root / 'traces-9b/candidates.jsonl').read_bytes()
        self.assertTrue(raw.startswith(saved))
        rows = [json.loads(s) for s in raw.splitlines()]
        self.assertEqual(rows, [{'sample': i, 'seed': 100+i, 'answer': i*i} for i in range(12)])
        self.assertEqual(read_json(self.root / 'launch.json')['attempt'], 2)

    def test_pending_pause_can_be_cancelled_without_new_inference(self):
        self.launch()
        old = read_json(self.root / 'launch.json')
        request_pause(self.root)
        result = resume(self.root)
        self.assertEqual(result['desired_state'], 'running')
        self.assertEqual(read_json(self.root / 'launch.json'), old)
        wait_until(lambda: read_json(self.root / 'pipeline-status.json').get('status') == 'complete')
        self.assertEqual(read_json(self.root / 'traces-9b/summary.json')['samples'], 12)

    def test_stale_pid_is_not_signalled_and_pause_survives_without_process(self):
        wrong = identity(psutil.Process())
        wrong['process_created'] += 100
        (self.root / 'launch.json').write_text(json.dumps(wrong))
        self.assertIsNone(matching_process(wrong))
        self.assertEqual(request_pause(self.root)['status'], 'paused')
        self.assertEqual(read_json(self.root / 'control.json')['desired_state'], 'paused')

    def test_orphaned_worker_prevents_second_inference(self):
        script = self.root / 'source/traces.py'
        script.write_text('import time\ntime.sleep(30)\n')
        orphan = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
        try:
            fake = {'pid': 99999999, 'process_created': 0}
            (self.root / 'launch.json').write_text(json.dumps(fake))
            with self.assertRaisesRegex(ValueError, 'Surviving inference child'):
                resume(self.root)
            attempted = subprocess.run([sys.executable, str(self.root / 'launch.py'), '--resume'],
                                       capture_output=True, text=True)
            self.assertNotEqual(attempted.returncode, 0)
            self.assertIn('still active', attempted.stderr)
        finally:
            orphan.terminate()
            orphan.wait(timeout=3)


class DashboardControlTests(unittest.TestCase):
    def test_intentional_pause_is_not_shown_as_a_crash(self):
        dashboard = Dashboard.__new__(Dashboard)
        dashboard.pipeline = {'status': 'incomplete', 'stage': 'teacher'}
        dashboard.control = {'desired_state': 'paused', 'status': 'paused'}
        dashboard.process_alive = False
        self.assertTrue(dashboard.state()[0].startswith('PAUSED'))
        dashboard.process_alive = True
        self.assertTrue(dashboard.state()[0].startswith('PAUSE MISMATCH'))


if __name__ == '__main__':
    unittest.main()
