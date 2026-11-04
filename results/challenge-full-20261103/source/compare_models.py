"""Paired, single-attempt Sudoku benchmark for Apple Silicon (MLX)."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from config import ModelConfig
from evaluation import evaluate_response
from prompts import prompts
from sudoku import build_prompt, random_solution, with_empty_cells

MODEL_IDS = {
    "2b": "Qwen/Qwen3.5-2B",
    "9b": "mlx-community/Qwen3.5-9B-4bit",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_tasks(count, empty_cells, seed):
    """Different solved boards per task; identical task list for every model."""
    base = prompts["sudoku9x9"]["hard"].data
    tasks = []
    seen = set()
    for blanks in empty_cells:
        for index in range(count):
            task_seed = seed + len(tasks)
            solution = random_solution(base, task_seed)
            data = with_empty_cells({**base, "expected_solution": solution}, blanks, task_seed)
            prompt = build_prompt(data)
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            if digest in seen:
                raise ValueError("Duplicate puzzle generated; choose another seed.")
            seen.add(digest)
            tasks.append({"id": digest, "empty_cells": blanks, "seed": task_seed,
                          "prompt": prompt, "data": data})
    return tasks


def summarize(rows, tasks, status):
    def rates(selected, planned):
        n = len(selected)
        passed = sum(row["passed"] for row in selected)
        return {"planned": planned, "completed": n, "correct": passed,
                "pass_at_1": passed / n if n else None,
                "mathematically_solved": sum(row["solved"] for row in selected),
                "format_failures": sum(not row["format_ok"] for row in selected),
                "token_limit_reached": sum(row["token_limit_reached"] for row in selected),
                "generation_seconds": sum(row["generation_seconds"] for row in selected)}
    return {"status": status, **rates(rows, len(tasks)), "by_empty_cells": {
        str(blanks): rates([r for r in rows if r["empty_cells"] == blanks],
                          sum(t["empty_cells"] == blanks for t in tasks))
        for blanks in sorted({task["empty_cells"] for task in tasks})}}


def run_worker(output, label, resume=False):
    # Heavy dependencies are loaded only in the child, never in the parent.
    from models import load_model
    import mlx.core as mx

    manifest = json.loads((output / "manifest.json").read_text())
    tasks = json.loads((output / "tasks.json").read_text())
    rows_path = output / f"{label}.jsonl"
    previous = []
    if resume and rows_path.exists():
        previous = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        expected_ids = [task["id"] for task in tasks[:len(previous)]]
        if ([row["task_id"] for row in previous] != expected_ids
                or any(row["model_id"] != MODEL_IDS[label] for row in previous)):
            raise ValueError("Saved responses are not a prefix of this model's task list.")
        if len(previous) == len(tasks):
            return
    config = ModelConfig(model_id=MODEL_IDS[label], backend="mlx",
                         thinking=manifest["thinking"],
                         max_output_tokens=manifest["max_output_tokens"],
                         max_prompt_tokens=manifest["max_prompt_tokens"])
    metadata = {"model_config": asdict(config), "python": sys.version,
                "previous_completed": len(previous),
                "packages": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")}}
    write_json(output / f"{label}.metadata.json", metadata)
    start = time.perf_counter()
    model = load_model(config)
    metadata["load_seconds"] = time.perf_counter() - start
    write_json(output / f"{label}.metadata.json", metadata)
    with rows_path.open("a" if resume else "x", encoding="utf-8") as handle:
        for index, task in enumerate(tasks, 1):
            if index <= len(previous):
                continue
            mx.reset_peak_memory()
            start = time.perf_counter()
            def progress(tokens, tps):
                print(f"{label}: case {index}/{len(tasks)}, {tokens} tokens, {tps:.1f} tok/s", flush=True)
            generation = manifest.get("sampling", {})
            result = model.generate_with_stats(task["prompt"], **generation,
                                               seed=(manifest["seed"] + index) % 2**32,
                                               progress_callback=progress)
            elapsed = time.perf_counter() - start
            report = evaluate_response(result.text, task["data"])
            row = {"task_id": task["id"], "model_id": config.model_id,
                   "empty_cells": task["empty_cells"], "completion": result.text,
                   "passed": report["passed"], "solved": report["solved"],
                   "format_ok": report["parsed"].format_ok,
                   "parser_issues": report["parsed"].issues, "checks": report["checks"],
                   "prompt_tokens": result.prompt_tokens, "generated_tokens": result.generated_tokens,
                   "finish_reason": result.finish_reason,
                   "token_limit_reached": result.generated_tokens >= config.max_output_tokens,
                   "generation_seconds": elapsed, "mlx_peak_bytes": mx.get_peak_memory()}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{label}: {index}/{len(tasks)} | vacíos={task['empty_cells']} | "
                  f"{'PASS' if report['passed'] else 'FAIL'} | {elapsed:.1f}s", flush=True)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=["2b", "9b"])
    parser.add_argument("--cases-per-level", type=positive, default=10)
    parser.add_argument("--empty-cells", nargs="+", type=int, default=[10, 20, 40])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-output-tokens", type=positive, default=8192)
    parser.add_argument("--max-prompt-tokens", type=positive, default=2048)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="save tasks without loading models")
    parser.add_argument("--worker", choices=MODEL_IDS, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help="resume --output using its saved manifest (other experiment flags are ignored)")
    args = parser.parse_args()
    if args.worker:
        if args.output is None:
            parser.error("worker requires --output")
        run_worker(args.output, args.worker, resume=args.resume)
        return 0
    if any(not 1 <= n <= 80 for n in args.empty_cells):
        parser.error("--empty-cells must be between 1 and 80 (generation may fail for high values)")
    if len(set(args.models)) != len(args.models) or len(set(args.empty_cells)) != len(args.empty_cells):
        parser.error("models and empty-cell counts must not repeat")
    if (not all(math.isfinite(x) for x in (args.temperature, args.top_p, args.presence_penalty))
            or args.temperature < 0 or not 0 < args.top_p <= 1 or args.top_k < 0
            or not 0 <= args.presence_penalty <= 2):
        parser.error("invalid sampling parameters")
    output = (args.output or Path("results") / datetime.now().strftime("compare-%Y%m%d-%H%M%S-%f")).resolve()
    if output.exists() and not args.resume:
        parser.error(f"output already exists: {output}; choose a new directory")
    if args.resume:
        if args.output is None:
            parser.error("--resume requires --output")
        manifest = json.loads((output / "manifest.json").read_text())
        tasks = json.loads((output / "tasks.json").read_text())
        args.models = list(manifest["models"])
        if any(MODEL_IDS.get(label) != repo for label, repo in manifest["models"].items()):
            parser.error("saved model IDs differ from the current model registry")
    else:
        tasks = make_tasks(args.cases_per_level, args.empty_cells, args.seed)
        output.mkdir(parents=True)
        write_json(output / "tasks.json", tasks)
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                    "models": {label: MODEL_IDS[label] for label in args.models},
                    "seed": args.seed, "cases_per_level": args.cases_per_level,
                    "empty_cells": args.empty_cells, "thinking": args.thinking,
                    "max_output_tokens": args.max_output_tokens,
                    "max_prompt_tokens": args.max_prompt_tokens,
                    "decoding": "greedy" if args.temperature == 0 else "sampled",
                    "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                                 "top_k": args.top_k, "presence_penalty": args.presence_penalty},
                    "generation_seed": "(dataset seed + one-based task index) modulo 2**32",
                    "shots": 0,
                    "note": "Empty-cell groups are not calibrated difficulty levels. One attempt per task. "
                            "2B uses original weights; 9B uses 4-bit weights. "
                            "Differences reflect both model size and weight precision."}
        write_json(output / "manifest.json", manifest)
    print(f"{len(tasks)} puzzles compartidos. Resultados: {output}", flush=True)
    if args.dry_run:
        return 0
    summaries = {}
    for label in args.models:
        print(f"\nCargando {MODEL_IDS[label]} (descarga en la primera ejecución)...", flush=True)
        # Process exit releases model weights and Metal allocations before the next model.
        with (output / f"{label}.stderr.log").open("w") as errors:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        "--worker", label, "--output", str(output),
                                        *(["--resume"] if args.resume else [])], stderr=errors)
        rows_path = output / f"{label}.jsonl"
        rows = []
        if rows_path.exists():
            for line in rows_path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # A killed worker may leave its last line incomplete.
        status = "complete" if completed.returncode == 0 and len(rows) == len(tasks) else "incomplete"
        summaries[label] = summarize(rows, tasks, status)
        write_json(output / "summary.json", summaries)
        print(f"{label}: {summaries[label]['correct']}/{len(rows)} aciertos; "
              f"{len(rows)}/{len(tasks)} completados ({status})", flush=True)
        if status != "complete":
            print(f"Revisar {output / (label + '.stderr.log')}. Corrida incompleta; no comparar tasas parciales.")
            return 1
    if "2b" in summaries and "9b" in summaries:
        delta = 100 * (summaries["9b"]["pass_at_1"] - summaries["2b"]["pass_at_1"])
        print(f"\nDiferencia observada 9B − 2B: {delta:+.1f} puntos porcentuales.")
        print("Resultado descriptivo de esta muestra, no prueba de superioridad general.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
