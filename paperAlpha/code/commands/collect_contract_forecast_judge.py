"""One real API forecast per permitted request-time prefix; no labels in prompts.

prepare is offline. run is one exclusive fixed cohort with no transport/JSON
retry or outcome-based replacement. Inputs are reconstructed post-acquisition,
so this is offline prefix replay, not an online latency/early-warning trial.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_submission_harnessaudit import read, sha, write_new, digest
from paperalpha_monitor.contract_forecast_llm import PROFILE, request_for, parse_forecast
from paperalpha_runtime.quadsentinel_adapter import create_api_client
from paperalpha_runtime.privacy_transfer import delivered_text

SOURCES = ('code/commands/collect_contract_forecast_judge.py', 'code/tests/test_contract_forecast_judge_collection.py',
           'code/src/paperalpha_monitor/contract_forecast_llm.py',
           'code/src/paperalpha_monitor/contract_forecast.py',
           'code/src/paperalpha_runtime/quadsentinel_adapter.py',
           'code/src/paperalpha_runtime/privacy_transfer.py')
API = {"endpoint": "http://localhost:58661/v1", "model": "gpt-5.5", "effort": "low",
       "timeout_seconds": 60, "transport_retries": 0, "parse_retries": 0, "concurrency": 2}


def schedule(public_episodes, contracts):
    """Every covered prefix; no future label or condition parameter exists."""
    rows = []
    seen = set()
    for episode in sorted(public_episodes, key=lambda x: x["episode_id"]):
        if episode["episode_id"] in seen:
            raise ValueError("duplicate public episode")
        seen.add(episode["episode_id"])
        claims = set()
        for window in episode["windows"]:
            number = window["claim_number"]
            if number in claims:
                raise ValueError("duplicate public request window")
            claims.add(number)
            if not window["covered"]:
                continue
            request = request_for(window["projection"], contracts[episode["task_id"]])
            rows.append({"episode_id": episode["episode_id"], "task_id": episode["task_id"],
                "claim_number": number, "current_a": window["projection"]["current_a"],
                "request": request, "request_sha256": digest(request)})
    return rows


def prepare(dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("judge cohort already exists")
    projection_report = read(dataset / "private/REPORT.json")
    if projection_report.get("status") != "projected":
        raise ValueError("completed request-time projection required")
    # Only artifact hashes are used; no actual target labels are consumed here.
    hashes = {name: value for name, value in projection_report["artifact_sha256"].items()
              if name.startswith("public/")}
    actual = {p.relative_to(dataset).as_posix() for p in (dataset / "public").rglob("*") if p.is_file()}
    if actual != set(hashes) or any(sha(dataset / name) != value for name, value in hashes.items()):
        raise ValueError("public predictor inventory changed")
    source_plan = read(dataset / "PLAN.json")
    episodes = [read(dataset / name) for name in sorted(hashes)]
    if {e["episode_id"] for e in episodes} != set(source_plan["episode_ids"]):
        raise ValueError("public cohort is not the full fixed episode set")
    rows = schedule(episodes, source_plan["contracts"])
    if not rows or len(rows) > 576:
        raise ValueError("nonempty bounded fixed request schedule required")
    output.mkdir(parents=True)
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    for name in SOURCES:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    plan = {"schema": PROFILE + "-collection", "dataset": str(dataset),
        "dataset_plan_sha256": sha(dataset / "PLAN.json"), "public_sha256": hashes,
        "source_sha256": source_hashes, "cases": rows, "maximum_api_calls": len(rows),
        "api": dict(API),
        "scope": "One fixed full-public-prefix joint forecast per covered actual next-reply boundary; offline replay, no intervention.",
        "labels_read": False, "test_tuning": False, "calibration": "Not claimed"}
    plan["sha256"] = digest(plan)
    write_new(output / "PLAN.json", plan)
    return {"prepared": True, "scheduled_api_calls": len(rows), "api_calls": 0}


def load_plan(output):
    output = Path(output).resolve()
    plan = read(output / "PLAN.json")
    if plan["sha256"] != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("judge plan changed")
    if (plan.get("schema") != PROFILE + "-collection" or plan.get("api") != API
            or set(plan["source_sha256"]) != set(SOURCES)
            or type(plan.get("maximum_api_calls")) is not int or not 0 < plan["maximum_api_calls"] <= 576):
        raise ValueError("fixed judge profile/source/budget required")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected or sha(output / "source" / name) != expected:
            raise ValueError("judge source changed")
    dataset = Path(plan["dataset"])
    if sha(dataset / "PLAN.json") != plan["dataset_plan_sha256"]:
        raise ValueError("dataset plan changed")
    actual = {p.relative_to(dataset).as_posix() for p in (dataset / "public").rglob("*") if p.is_file()}
    if actual != set(plan["public_sha256"]) or any(sha(dataset / n) != h for n, h in plan["public_sha256"].items()):
        raise ValueError("public prefix data changed")
    if len(plan["cases"]) != plan["maximum_api_calls"]:
        raise ValueError("API schedule denominator changed")
    source_plan = read(dataset / "PLAN.json")
    episodes = [read(dataset / name) for name in sorted(plan["public_sha256"])]
    if (plan["cases"] != schedule(episodes, source_plan["contracts"])
            or any(c["request_sha256"] != digest(c["request"]) for c in plan["cases"])):
        raise ValueError("judge schedule differs from unchanged public inputs")
    return plan


async def run(output):
    output = Path(output).resolve()
    plan = load_plan(output)
    key = os.environ.get("PAPERALPHA_LOCAL_API_KEY")
    if not key:
        raise ValueError("transient environment credential required")
    write_new(output / "RUN.claim.json", {"plan_sha256": plan["sha256"], "status": "exclusively_claimed"})
    api = create_api_client(plan["api"]["endpoint"], key)
    semaphore = asyncio.Semaphore(plan["api"]["concurrency"])
    async def one(index, case):
        async with semaphore:
            folder = output / "runs" / f"{index:04d}"
            folder.mkdir(parents=True, exist_ok=False)
            started = time.monotonic_ns()
            write_new(folder / "REQUEST.json", {"plan_sha256": plan["sha256"], **case, "monotonic_ns": started})
            record = {"episode_id": case["episode_id"], "task_id": case["task_id"],
                "claim_number": case["claim_number"], "index": index, "status": "failed", "error_type": None}
            try:
                raw = (await api.responses.create(**case["request"])).model_dump(mode="json")
                response = {k: raw.get(k) for k in ("id", "model", "status", "incomplete_details", "usage", "error")}
                response["output"] = [m for m in raw.get("output", []) if m.get("type") == "message"]
                write_new(folder / "RESPONSE.json", response)
                record["sdk_response_id"] = response["id"]
                if not isinstance(response["id"], str) or not response["id"].strip() or response["error"] is not None:
                    raise ValueError("nonempty response identity without explicit API error required")
                text = delivered_text(response)
                if text["native_refusal"]:
                    raise ValueError("refusal is not a forecast")
                record["prediction"] = parse_forecast(text["text"], case["current_a"])
                record["status"] = "predicted"
            except Exception as exc:
                record["error_type"] = type(exc).__name__
            record["elapsed_ns"] = time.monotonic_ns()-started
            write_new(folder / "RESULT.json", record)
            return record
    try:
        records = await asyncio.gather(*(one(i, c) for i, c in enumerate(plan["cases"], 1)))
    finally:
        await api.close()
    load_plan(output)
    identities = Counter(r["sdk_response_id"] for r in records
                         if isinstance(r.get("sdk_response_id"), str) and r["sdk_response_id"].strip())
    duplicates = {k for k, n in identities.items() if n > 1}
    for record in records:
        if isinstance(record.get("sdk_response_id"), str) and record["sdk_response_id"] in duplicates:
            record.update(status="failed", error_type="DuplicateResponseIdentity")
            record.pop("prediction", None)
    write_new(output / "IDENTITY_AUDIT.json", {"duplicate_response_ids": sorted(duplicates),
        "affected_indices": [r["index"] for r in records
                             if isinstance(r.get("sdk_response_id"), str) and r["sdk_response_id"] in duplicates],
        "semantics": "Collection report excludes every duplicated identity; immutable per-call RESULT is the preliminary parse, not final provenance acceptance."})
    status = dict(Counter(r["status"] for r in records))
    report = {"schema": PROFILE + "-collection-report", "plan_sha256": plan["sha256"],
        "status": "completed" if status.get("failed", 0) == 0 else "completed_with_failures",
        "api_attempts": len(records), "outcomes": status, "records": records,
        "performance_evaluated": False, "labels_read": False,
        "artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}}
    write_new(output / "REPORT.json", report)
    return {k: report[k] for k in ("status", "api_attempts", "outcomes")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "verify"))
    parser.add_argument("--dataset", type=Path, default=ROOT / 'experiments/harnessaudit/runs/submission__harnessaudit_paired/forecast')
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/harnessaudit/runs/submission__harnessaudit_paired/llm_judge')
    args = parser.parse_args()
    result = prepare(args.dataset, args.out) if args.mode == "prepare" else (
        asyncio.run(run(args.out)) if args.mode == "run" else {"verified": bool(load_plan(args.out))})
    print(json.dumps(result))
