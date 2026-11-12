"""Freeze a metadata-selected acquisition cohort before any model outcomes.

This is a development collection, not a fixed-DAG or held-out performance test.
Run each original pilot in a fresh process. Never replace failed/positive runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'code/external/HarnessAudit'
REVISION = "6317162590aeeb1c8dde32b880ac199933343e4a"
SEED = "paperalpha-development-cohort-20260919-v1"
EXCLUDED = ("dl-t11", "dl-t14")
CATEGORIES = ("dining", "housing", "travel", "wellness")
SOURCES = ('code/commands/harnessaudit_cohort.py', 'code/commands/collect_harnessaudit_api_pilot.py',
           'code/commands/audit_harnessaudit_api_pilot.py', 'code/src/paperalpha_runtime/journal.py',
           'code/src/paperalpha_monitor/harnessaudit_observations.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def inventory(upstream):
    result, seen = [], set()
    for path in sorted((upstream / "multi_agent/tasks/daily_life").rglob("*.yaml")):
        task = yaml.safe_load(path.read_text(encoding="utf-8"))
        identity = task["task_id"]
        if not re.fullmatch(r"dl-t[0-9]+", identity) or identity in seen or path.stem != identity:
            raise ValueError("unique task IDs matching filenames required")
        seen.add(identity)
        category = task["category"]
        if category not in CATEGORIES or path.parent.name != category:
            raise ValueError("unexpected category layout")
        reasons = []
        if identity in EXCLUDED:
            reasons.append("previous_development_task")
        if (task.get("domain") != "daily_life" or task.get("modality") != "text_only"
                or task.get("input_assets") or task.get("fixture")):
            reasons.append("outside_reviewed_text_only_sqlite_scope")
        result.append({"task_id": identity, "category": category,
                       "task_path": path.relative_to(upstream).as_posix(), "task_sha256": sha(path),
                       "eligible": not reasons, "exclusion_reasons": reasons})
    return result


def select_tasks(candidates, per_category=1):
    if type(per_category) is not int or not 1 <= per_category <= 2:
        raise ValueError("one or two tasks per category required")
    groups = defaultdict(list)
    for task in candidates:
        if task["eligible"]:
            groups[task["category"]].append(task)
    selected = []
    for category in CATEGORIES:
        ordered = sorted(groups[category], key=lambda t: (digest([SEED, t["task_id"]]), t["task_id"]))
        if len(ordered) < per_category:
            raise ValueError("insufficient eligible tasks; do not substitute after results")
        selected.extend(ordered[:per_category])
    return selected


def check_upstream(upstream):
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    if revision != REVISION or dirty:
        raise ValueError("unchanged reviewed upstream required")


def prepare(output, upstream=UPSTREAM, per_category=1):
    output, upstream = output.resolve(), upstream.resolve()
    if output.exists():
        raise ValueError("new cohort directory required")
    check_upstream(upstream)
    candidates = inventory(upstream)
    chosen = select_tasks(candidates, per_category)
    plan = {"schema": "harnessaudit-development-cohort-v1", "created_at": now(),
            "purpose": "New API-generated development observations, not a final test or performance result.",
            "upstream": str(upstream), "upstream_commit": REVISION, "selection_seed": SEED,
            "selection": "lowest SHA256(seed, task_id) per category; metadata only; no outcome input",
            "excluded_previous_tasks": list(EXCLUDED), "inventory": candidates, "tasks": chosen,
            "settings": {"model": "gpt-5.5", "request_budget": 32, "max_turns": 8, "timeout": 300},
            "max_total_requests": len(chosen) * 32, "attempts_per_task": 1,
            "source_sha256": {name: sha(ROOT / name) for name in SOURCES},
            "protocol": {
                "target": "any_recorded_access_rule_violation", "failed_run_label": None,
                "labels": "Recompute official rule checks after execution; never feed decisions to monitor.",
                "zero_meaning": "No recorded rule match, not proven safety or task success.",
                "split": "Unsplit development acquisition. Freeze a later training/evaluation protocol before fitting.",
                "retention": "Keep every planned attempt, including failures; no replacement or class balancing by deletion.",
                "topology": "Official hub/spoke returns allowed; not a fixed-DAG benchmark.",
                "missing": "No actual-harm timestamp, independent mechanism labels, authenticated checks or guarantees.",
                "negative_requirement": "Single-class outcomes cannot support AUROC/FPR or train a discriminative final-outcome model."
            }}
    plan["plan_sha256"] = digest(plan)
    output.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        dest = output / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle:
            handle.write((ROOT / name).read_bytes())
    write_new(output / "PLAN.json", plan)
    return plan


def load_plan(path, verify_sources=True):
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "harnessaudit-development-cohort-v1":
        raise ValueError("unsupported plan")
    expected = plan.get("plan_sha256")
    if digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != expected:
        raise ValueError("plan content changed")
    if verify_sources:
        for name, expected in plan["source_sha256"].items():
            if sha(ROOT / name) != expected or sha(path.parent / "source_snapshot" / name) != expected:
                raise ValueError(f"acquisition source changed: {name}")
    return plan


def run_task(plan_path, identity):
    plan_path = plan_path.resolve()
    plan = load_plan(plan_path)
    matches = [t for t in plan["tasks"] if t["task_id"] == identity]
    if len(matches) != 1:
        raise ValueError("task is not in the frozen cohort")
    task, upstream = matches[0], Path(plan["upstream"])
    check_upstream(upstream)
    if sha(upstream / task["task_path"]) != task["task_sha256"]:
        raise ValueError("task changed after cohort freeze")
    output = plan_path.parent / "runs" / identity
    if output.exists():
        raise ValueError("task was already attempted; no automatic retry or overwrite")
    settings = plan["settings"]
    command = [sys.executable, "-X", "utf8", "-B", str(ROOT / 'code/commands/collect_harnessaudit_api_pilot.py'),
               "--upstream", str(upstream), "--task", str(upstream / task["task_path"]),
               "--out", str(output), "--model", settings["model"], "--request-budget", str(settings["request_budget"]),
               "--max-turns", str(settings["max_turns"]), "--timeout", str(settings["timeout"])]
    # Exclusive claim prevents two wrappers from launching the same task.
    write_new(plan_path.parent / "attempts" / f"{identity}.started.json",
              {"task_id": identity, "plan_sha256": plan["plan_sha256"], "started_at": now(), "command": command})
    process = subprocess.run(command, check=False)
    receipt = {"task_id": identity, "plan_sha256": plan["plan_sha256"], "completed_at": now(),
               "returncode": process.returncode, "run_manifest_sha256": sha(output / "RUN.json") if (output / "RUN.json").exists() else None}
    write_new(plan_path.parent / "attempts" / f"{identity}.finished.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--per-category", type=int, default=1)
    run = sub.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--task", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        plan = prepare(args.out, per_category=args.per_category)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "tasks": [t["task_id"] for t in plan["tasks"]],
                          "max_total_requests": plan["max_total_requests"]}))
    else:
        print(json.dumps(run_task(args.plan, args.task)))


if __name__ == "__main__":
    main()
