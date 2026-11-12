"""Separately frozen partial-observation extension of the repeated study.

Prepare does not collect or score. Evaluation invokes the RAW collection's
unchanged frozen verifier in a separate process, then fits the new extension.
The original collection and its complete-pair protocol are never rewritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src')]
from paperalpha_monitor.observable_event_dag import ObservableEventDAG
from paperalpha_monitor.partial_repeated_message_evaluation import evaluate_partial_collection

SCRIPT = 'code/commands/evaluate_partial_repeated_messages.py'
TESTS = ('code/tests/test_partial_grouped_event_learning.py',
         'code/tests/test_partial_repeated_message_evaluation.py',
         'code/tests/test_partial_grouped_independent.py',
         'code/tests/test_partial_repeated_runner.py')
FIXTURE_DEPENDENCIES = (
    'code/commands/collect_repeated_message_study.py', 'code/commands/evaluate_repeated_messages.py',
    'code/commands/diagnose_repeated_messages.py', 'code/tests/test_repeated_message_study.py',
    'code/tests/test_repeated_message_evaluation.py', 'code/tests/test_repeated_message_diagnostics.py',
    'code/tests/test_grouped_event_learning.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def environment(stage):
    return {"stage": stage, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "sys_executable": str(Path(sys.executable).resolve()), "sys_executable_sha256": sha(sys.executable),
            "sys_version": sys.version, "platform": platform.platform(),
            "packages": {n: metadata.version(n) for n in ("numpy", "scipy", "scikit-learn")}}


def prepare(raw, out):
    raw, out = Path(raw).resolve(), Path(out).resolve()
    if out == raw or raw in out.parents or out in raw.parents:
        raise ValueError("extension must be outside and separate from the raw collection")
    if out.exists():
        raise ValueError("new extension directory required")
    raw_plan = read(raw / "PLAN.json")
    if raw_plan.get("schema") != "repeated-message-development-v1":
        raise ValueError("original repeated-message development plan required")
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'code/src').rglob("*.py")]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'knowledge/templates').glob("*.json")]
    names += ['knowledge/risks/risk_catalog.json', SCRIPT, *TESTS, *FIXTURE_DEPENDENCIES]
    # Check the complete inventory before creating an incomplete preparation.
    for name in names:
        if not (ROOT / name).is_file():
            raise ValueError("missing extension source: " + name)
    out.mkdir(parents=True, exist_ok=False)
    for name in sorted(set(names)):
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write(out / "PREPARE_ENVIRONMENT.json", environment("prepare_partial_extension"))
    plan = {"schema": "partial-repeated-message-extension-v1", "development_only": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_collection": str(raw), "raw_plan_sha256": sha(raw / "PLAN.json"),
            "prepared_before_collection": not (raw / "COLLECTION.json").exists(),
            "source_sha256": {n: sha(out / "source" / n) for n in sorted(set(names))},
            "prepare_environment_sha256": sha(out / "PREPARE_ENVIRONMENT.json"),
            "training_modes": ["complete_pairs", "all_observed"], "penalties": [.001, .03],
            "folds": "Leave one of the four declared base tasks out; all layouts/repeats stay together.",
            "normalization": "Both modes divide by all 36 planned training upstream groups, including empty and identity-excluded groups.",
            "scoring": "The same identity-qualified complete publisher pairs in both modes and all methods.",
            "missingness": "Missing/refused/incomplete/unattempted content is not zero. Observed-content likelihood assumes ignorable selection; no failure model or correction of MNAR bias.",
            "scope": "Optional development extension, not a rewrite of the original complete-pair analysis or an independent test.",
            "new_api_calls": 0, "selected_penalty": None}
    write(out / "PLAN.json", plan)
    return {"prepared": str(out), "plan_sha256": sha(out / "PLAN.json"), "new_api_calls": 0}


def source_evidence(out):
    if ROOT.resolve() != (out / "source").resolve():
        raise ValueError("run the extension from its own frozen source")
    plan = read(out / "PLAN.json")
    if plan.get("schema") != "partial-repeated-message-extension-v1":
        raise ValueError("partial extension plan required")
    expected = plan["source_sha256"]
    actual_names = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'code/src').rglob("*.py")}
    if actual_names != {n for n in expected if n.startswith('code/src/') and n.endswith(".py")}:
        raise ValueError("extension Python inventory changed")
    actual = {n: sha(ROOT / n) for n in expected}
    if actual != expected or SCRIPT not in expected or sha(__file__) != expected[SCRIPT]:
        raise ValueError("extension frozen source changed")
    if sha(out / "PREPARE_ENVIRONMENT.json") != plan["prepare_environment_sha256"]:
        raise ValueError("preparation environment changed")
    loaded = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith(("paperalpha_monitor.", "paperalpha_runtime.")):
            continue
        key = 'code/src/' + name.replace(".", "/") + ".py"
        path = getattr(module, "__file__", None)
        if path is None or key not in expected or sha(path) != expected[key]:
            raise ValueError("loaded extension module differs from frozen source: " + name)
        loaded[key] = sha(path)
    return {"source_sha256": actual, "loaded_modules_sha256": loaded}


def frozen_raw_verify(raw):
    """Use the original source tree, not the newly extended module imports."""
    script = raw / "source/code/commands" / "collect_repeated_message_study.py"
    plan = read(raw / "PLAN.json")
    if sha(script) != plan["source_sha256"]['code/commands/collect_repeated_message_study.py']:
        raise ValueError("original frozen verifier changed")
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", str(script), "verify", "--out", str(raw)],
                            cwd=str(raw / "source"), text=True, encoding="utf-8", capture_output=True,
                            timeout=180, check=False)
    if result.returncode:
        raise ValueError("original frozen verifier failed: " + result.stderr[-2000:])
    value = json.loads(result.stdout)
    if value.get("schema") != "verified-repeated-message-development-v1":
        raise ValueError("unexpected frozen verification response")
    fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                           allow_nan=False).encode("utf-8")).hexdigest()
    for group in value["grouped_traces"]:
        group["dag"] = ObservableEventDAG(**group["dag"])
    return value, fingerprint


def evaluate(out, *, max_iter=500):
    out = Path(out).resolve()
    if (out / "REPORT.json").exists() or (out / "FIT_ENVIRONMENT.json").exists():
        raise ValueError("extension evaluation already started; never overwrite it")
    source_before = source_evidence(out)
    plan = read(out / "PLAN.json")
    raw = Path(plan["raw_collection"]).resolve()
    if sha(raw / "PLAN.json") != plan["raw_plan_sha256"]:
        raise ValueError("bound original collection plan changed")
    original = {n: sha(raw / n) for n in ("PLAN.json", "COLLECTION.json")}
    verified, verified_sha = frozen_raw_verify(raw)
    write(out / "FIT_ENVIRONMENT.json", environment("fit_partial_extension"))
    report = evaluate_partial_collection(verified, max_iter=max_iter)
    _, after_sha = frozen_raw_verify(raw)
    source_after = source_evidence(out)
    if original != {n: sha(raw / n) for n in original} or verified_sha != after_sha:
        raise ValueError("original collection changed during fitting")
    if source_before["source_sha256"] != source_after["source_sha256"]:
        raise ValueError("extension source changed during fitting")
    report["execution"] = {"raw_collection": str(raw), "input_sha256": original,
                           "verified_collection_sha256": verified_sha, "plan_sha256": sha(out / "PLAN.json"),
                           "fit_environment_sha256": sha(out / "FIT_ENVIRONMENT.json"), "max_iter": max_iter,
                           "source_before": source_before, "source_after": source_after,
                           "new_api_calls": 0}
    write(out / "REPORT.json", report)
    return {"output": str(out / "REPORT.json"), "sha256": sha(out / "REPORT.json"),
            "predictions": len(report["predictions"]), "new_api_calls": 0, "development_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.collection is None:
            parser.error("prepare requires --collection")
        result = prepare(args.collection, args.out)
    else:
        if args.collection is not None:
            parser.error("evaluate uses the collection bound by the prepared plan")
        result = evaluate(args.out, max_iter=args.max_iter)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
