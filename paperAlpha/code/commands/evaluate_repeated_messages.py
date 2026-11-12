"""Read-only collector replay followed by offline four-fold development scoring.

All reports go to a new explicit directory outside the raw collection. No API,
collection retries, old tests, threshold selection or best-penalty selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_repeated_message_study import verify
from paperalpha_monitor.repeated_message_evaluation import evaluate_collection


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_evidence(raw):
    """Require the collection's frozen runner, complete source set and hashes."""
    if ROOT.resolve() != (raw/"source").resolve():
        raise ValueError("run this evaluator from the collection's frozen source")
    plan = json.loads((raw/"PLAN.json").read_text(encoding="utf-8"))
    expected = plan["source_sha256"]
    source_names = {str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'code/src').rglob("*.py")}
    if source_names != {n for n in expected if n.startswith('code/src/') and n.endswith(".py")}:
        raise ValueError("frozen Python source inventory mismatch")
    required = source_names | {'code/commands/collect_repeated_message_study.py', 'code/commands/evaluate_repeated_messages.py'}
    if not required <= set(expected):
        raise ValueError("collector and evaluator must both belong to the predeclared frozen source")
    actual = {name: sha(ROOT/name) for name in sorted(required)}
    if any(actual[n] != expected[n] for n in actual):
        raise ValueError("frozen evaluation source changed")
    # Also check the loaded modules, including direct function use in tests.
    # A different filesystem location is only acceptable when bytes match the
    # declared snapshot; the CLI root itself must still be the frozen runner.
    loaded = {}
    for name, module in list(sys.modules.items()):
        if name.startswith(("paperalpha_monitor.", "paperalpha_runtime.")):
            key = 'code/src/'+name.replace(".", "/")+".py"
        elif name == "collect_repeated_message_study":
            key = 'code/commands/collect_repeated_message_study.py'
        else:
            continue
        path = getattr(module, "__file__", None)
        if path is None or key not in expected or sha(path) != expected[key]:
            raise ValueError("loaded module is not the declared frozen code: " + name)
        loaded[key] = sha(path)
    if sha(__file__) != expected['code/commands/evaluate_repeated_messages.py']:
        raise ValueError("loaded evaluator differs from the declared frozen code")
    return {"source_sha256": actual, "loaded_modules_sha256": loaded}


def fit_environment():
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "offline_fit_and_evaluate",
            "sys_executable": str(Path(sys.executable).resolve()), "sys_executable_sha256": sha(sys.executable),
            "sys_version": sys.version, "platform": platform.platform(),
            "packages": {name: metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}}


def evaluate(raw, out, *, max_iter=500):
    raw, out = Path(raw).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("output must be a new directory; never overwrite an evaluation")
    if out == raw or raw in out.parents or out in raw.parents:
        raise ValueError("evaluation output must be outside and separate from the raw collection")
    source_before = source_evidence(raw)
    environment = fit_environment()
    original = {name: sha(raw/name) for name in ("PLAN.json", "COLLECTION.json")}
    # Collector.verify checks every frozen source and raw group file before fit.
    verified = verify(raw)
    report = evaluate_collection(verified, max_iter=max_iter)
    if original != {name: sha(raw/name) for name in original}:
        raise ValueError("raw collection identity changed during evaluation")
    # A second read-only replay detects changed raw files, not just manifests.
    verify(raw)
    source_after = source_evidence(raw)
    if source_before["source_sha256"] != source_after["source_sha256"]:
        raise ValueError("evaluation code changed while fitting")
    out.mkdir(parents=True, exist_ok=False)
    report["execution"] = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_collection": str(raw), "input_sha256": original, "max_iter": max_iter,
        "fit_environment": environment, "source_before": source_before, "source_after": source_after}
    with (out/"REPORT.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return {"output": str(out/"REPORT.json"), "sha256": sha(out/"REPORT.json"),
            "development_only": True, "predictions": len(report["predictions"]), "coverage": report["coverage"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.collection, args.out, max_iter=args.max_iter), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
