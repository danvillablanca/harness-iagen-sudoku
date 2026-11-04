"""Shared persistence and provenance for resumable inference experiments."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sample_seed(seed, task_id, index):
    return int.from_bytes(hashlib.sha256(json.dumps([seed, task_id, index]).encode()).digest()[:4], 'big')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def load_samples(path, *, repair_tail=False):
    """Recover only a torn final write; never silently ignore corrupt full rows."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows, offset = [], 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if repair_tail and i == len(lines) - 1 and not line.endswith(b'\n'):
                backup = path.with_suffix(path.suffix + '.partial')
                with backup.open('xb') as handle:
                    handle.write(line)
                path.write_bytes(data[:offset])
                return rows
            raise ValueError(f'Invalid JSON at {path}:{i + 1}') from None
        rows.append(row)
        offset += len(line)
    # A valid JSON record without a final newline is also safe to resume.
    if repair_tail and data and not data.endswith(b'\n'):
        with path.open('ab') as handle:
            handle.write(b'\n')
    return rows


def provenance(model_id, runner):
    packages = {}
    for name in ('torch', 'transformers', 'mlx', 'mlx-lm', 'huggingface-hub'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(model_id, 'config.json')
    revision = Path(cached).parent.name if isinstance(cached, str) else None
    root = Path(__file__).parent
    return {
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': packages, 'cached_model_revision': revision,
        'git_commit': "dd1e2956880022eecbb93a765414ce58f4524769",
        'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [root / name for name in ('qwen.py', 'models.py', 'config.py',
                            'sudoku.py', 'response_parser.py', 'evaluation.py', 'reward.py',
                            'experiment.py', 'baseline.py', runner)]},
    }


def prepare_manifest(path, specification, *, resume=False):
    """Bind a run to its dataset, decoding settings and executable code."""
    path = Path(path)
    manifest = {**specification, 'run_id': digest(specification)}
    if path.exists():
        if not resume:
            raise ValueError(f'{path} already exists; use --resume with identical arguments.')
        previous = json.loads(path.read_text())
        if previous != manifest:
            raise ValueError('Resume settings, input data, code or environment changed; use a new output.')
        return previous
    if resume:
        raise ValueError(f'Cannot resume: missing {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    return manifest
