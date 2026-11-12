"""Freeze, collect, and read-only verify grouped API-message development data.

No fitting or model scoring. CLI collect/verify run from the frozen source.
All output files use exclusive creation: observation timeouts are not restarts.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime import repeated_message_study as study
from paperalpha_runtime.privacy_transfer import delivered_text


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def runtime_environment(stage):
    """Narrow allowlist only; never enumerate environment variables or secrets."""
    if stage not in {"prepare", "collect"}:
        raise ValueError("known execution stage required")
    packages = {}
    for name in ("numpy", "scipy", "openai", "httpx"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    executable = str(Path(sys.executable).resolve()) if sys.executable else None
    try:
        executable_hash = sha(executable) if executable else None
    except OSError:
        executable_hash = None
    return {"schema": "repeated-message-runtime-environment-v1", "stage": stage,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "sys_executable": executable, "sys_executable_sha256": executable_hash,
            "sys_version": sys.version,
            "platform": {"system": platform.system(), "release": platform.release(),
                         "version": platform.version(), "machine": platform.machine(),
                         "python_implementation": platform.python_implementation()},
            "packages": packages}


def check_environment(path, expected_hash, stage):
    if sha(path) != expected_hash:
        raise ValueError("execution environment record changed")
    value = read(path)
    if (set(value) != {"schema", "stage", "captured_at_utc", "sys_executable", "sys_executable_sha256", "sys_version", "platform", "packages"}
            or value["schema"] != "repeated-message-runtime-environment-v1" or value["stage"] != stage
            or set(value["packages"]) != {"numpy", "scipy", "openai", "httpx"}
            or any(v is not None and not isinstance(v, str) for v in value["packages"].values())
            or set(value["platform"]) != {"system", "release", "version", "machine", "python_implementation"}):
        raise ValueError("invalid execution environment scope")
    # These are historical observations. Do not replace them with, or require
    # equality to, the Python/host currently running this read-only verifier.
    return value


def prepare(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    names = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'code/src').rglob("*.py")]
    names += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'knowledge/templates').glob("*.json")]
    names += ['knowledge/risks/risk_catalog.json', 'code/commands/collect_repeated_message_study.py',
              'code/commands/diagnose_repeated_messages.py', 'code/tests/test_repeated_message_diagnostics.py',
              'code/commands/evaluate_repeated_messages.py', 'code/tests/test_repeated_message_evaluation.py',
              'code/tests/test_repeated_message_study.py', 'code/tests/test_grouped_event_learning.py']
    for name in sorted(set(names)):
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write(out / "PREPARE_ENVIRONMENT.json", runtime_environment("prepare"))
    plan = {"schema": "repeated-message-development-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cases": study.cases(), "base_tasks": study.base_tasks(),
            "prepare_environment_sha256": sha(out / "PREPARE_ENVIRONMENT.json"),
            "source_sha256": {n: sha(out / "source" / n) for n in sorted(set(names))},
            "api": copy.deepcopy(study.API_SETTINGS),
            "planned_groups": 48, "planned_continuation_pairs": 96, "maximum_api_calls": study.API_CALL_BUDGET,
            "development_only": True, "predictions": None, "new_fits": 0, "old_test_read": False,
            "failure_protocol": "No retries or replacement groups. Stop group after upstream failure; attempt both publishers of each continuation after complete upstream. Keep every failure; never make a negative label from failure.",
            "training_protocol": "One early DAG per upstream group; reviewer labels counted once, each complete publisher pair is one continuation. No fitting when no complete continuation. Partial-completion exclusion can bias training; report it.",
            "response_identity_protocol": "Audit all provider responses whose status is completed, including completed refusals. Require nonempty unique provider response IDs across the entire collection and returned model exactly gpt-5.5; no undeclared alias or dated snapshot accepted. Exclude every involved group from statistics and fitting on missing/duplicate IDs or missing/different model, but preserve all raw records and failures. Identical text is not an exclusion. Unique IDs alone do not prove statistical independence.",
            "grouping": "Keep all repetitions, layouts and continuations of each base_task in the same future split; no independent test split is declared here.",
            "scope": "Four fictional canary tasks, not semantic leakage, actual harm, a calibrated safety system, or prospective performance. Output variation and positive counts are not guaranteed.",
            "openai_docs": study.DOCS}
    if sum(study.planned_calls(c) for c in plan["cases"]) != study.API_CALL_BUDGET:
        raise ValueError("fixed API budget mismatch")
    write(out / "PLAN.json", plan)
    return plan


def load_plan(out):
    out = Path(out).resolve()
    plan = read(out / "PLAN.json")
    if plan.get("schema") != "repeated-message-development-v1" or plan["cases"] != study.cases() or plan["base_tasks"] != study.base_tasks():
        raise ValueError("fixed development protocol changed")
    if (plan["maximum_api_calls"] != 264 or plan["predictions"] is not None or not plan["development_only"]
            or plan["api"] != study.API_SETTINGS or plan["planned_groups"] != 48 or plan["planned_continuation_pairs"] != 96):
        raise ValueError("not the unscored bounded development protocol")
    for name, expected in plan["source_sha256"].items():
        if sha(out / "source" / name) != expected:
            raise ValueError("frozen source changed: " + name)
    check_environment(out / "PREPARE_ENVIRONMENT.json", plan["prepare_environment_sha256"], "prepare")
    return plan


def visible_response(raw):
    return {**{k: raw.get(k) for k in ("id", "model", "status", "incomplete_details", "usage")},
            "output": [m for m in raw.get("output", []) if m.get("type") == "message"]}


def response_text(response):
    value = delivered_text(response)
    if value["native_refusal"]:
        raise ValueError("native refusal is a failed continuation, not a negative label")
    return value["text"]


def audit_response_identity(groups, *, scope):
    """Qualification over supplied groups, without deleting or relabeling data.

Single-group verification cannot certify cross-group uniqueness. Collection
verification always reruns this check over its complete planned queue.
"""
    if scope not in {"within_group_only", "complete_collection"}:
        raise ValueError("explicit identity audit scope required")
    expected_model = study.API_SETTINGS["model"]
    by_id, missing, model_mismatches = {}, [], []
    lookup = {group["id"]: group for group in groups}
    if len(lookup) != len(groups):
        raise ValueError("duplicate group IDs in identity audit")
    for group in groups:
        group["response_identity_issues"] = []
        for receipt in group["completed_response_receipts"]:
            evidence = {"group_id": group["id"], **receipt}
            identity = receipt["provider_response_id"]
            if not isinstance(identity, str) or not identity.strip():
                issue = {"reason": "missing_provider_response_id", **evidence}
                missing.append(issue)
                group["response_identity_issues"].append(issue)
            else:
                by_id.setdefault(identity, []).append(evidence)
            if receipt["returned_model"] != expected_model:
                issue = {"reason": "returned_model_not_declared", "expected_model": expected_model, **evidence}
                model_mismatches.append(issue)
                group["response_identity_issues"].append(issue)
    duplicate = []
    for identity, receipts in sorted(by_id.items()):
        if len(receipts) > 1:
            duplicate.append({"provider_response_id": identity, "records": receipts})
            for receipt in receipts:
                lookup[receipt["group_id"]]["response_identity_issues"].append(
                    {"reason": "duplicate_provider_response_id", **receipt})
    for group in groups:
        qualified = not group["response_identity_issues"]
        group["response_identity_qualified"] = qualified
        group["response_identity_audit_scope"] = scope
        group["eligible_for_grouped_fit"] = bool(group["upstream_complete"] and group["continuations"] and qualified)
        group["eligible_for_statistics"] = group["eligible_for_grouped_fit"]
    return {"scope": scope, "expected_returned_model": expected_model,
            "completed_response_records": sum(len(g["completed_response_receipts"]) for g in groups),
            "distinct_nonempty_provider_ids": len(by_id), "missing_provider_id_records": missing,
            "duplicate_provider_ids": duplicate, "model_mismatch_records": model_mismatches,
            "excluded_group_ids": [g["id"] for g in groups if not g["response_identity_qualified"]],
            "qualified_group_count": sum(g["response_identity_qualified"] for g in groups),
            "scope_warning": "Provider ID uniqueness is necessary provenance evidence, not proof of independent sampling. Text equality is not tested. Exact gpt-5.5 only; undeclared aliases/snapshots are not presumed equivalent."}


async def collect_group(out, case, api):
    out = Path(out).resolve()
    directory = out / "runs" / case["id"]
    directory.mkdir(parents=True, exist_ok=False)
    records, failures = [], []

    def record(kind, **value):
        sequence = len(records) + 1
        path = directory / f"record_{sequence:04d}.json"
        result = {"record_id": path.as_posix(), "sequence": sequence, "monotonic_ns": time.monotonic_ns(),
                  "at_utc": datetime.now(timezone.utc).isoformat(), "kind": kind, **value}
        write(path, result)
        records.append(result)
        return result

    async def call(node, messages, continuation):
        req = record("request", node=node, continuation=continuation, request=study.request(case, node, messages))
        try:
            raw = await api.responses.create(**req["request"])
        except Exception as exc:
            failure = record("failure", node=node, continuation=continuation, request_id=req["record_id"],
                             failure_kind="transport", error_type=type(exc).__name__, status_code=getattr(exc, "status_code", None))
            failures.append(failure["record_id"])
            return None
        raw = raw.model_dump(mode="json")
        res = record("response", node=node, continuation=continuation, request_id=req["record_id"], response=visible_response(raw))
        try:
            text = response_text(res["response"])
        except (ValueError, KeyError, TypeError) as exc:
            failure = record("failure", node=node, continuation=continuation, request_id=req["record_id"],
                             failure_kind="invalid_response", error_type=type(exc).__name__)
            failures.append(failure["record_id"])
            return None
        return {"record_id": res["record_id"], "sequence": res["sequence"], "text": text}

    source = record("source", text=case["source"])
    completed = {"source": {k: source[k] for k in ("record_id", "sequence", "text")}}
    messages = {"source": case["source"]}
    early = study.prefix(case, completed, len(records) + 1)
    record("prefix", phase="early", continuation=None, before_node=None, packet=early)
    write(directory / "EARLY.json", early)
    shared = {}
    for node in study.reviewers(case):
        reply = await call(node, messages, None)
        if reply is None:
            break
        completed[node], messages[node] = reply, reply["text"]
        shared[node] = study.label(case, reply)
    upstream_complete = set(shared) == set(study.reviewers(case))
    write(directory / "SHARED.json", {"completed": upstream_complete, "labels": shared})
    continuations = []
    if upstream_complete:
        for continuation in range(study.CONTINUATIONS):
            local = dict(completed)
            labels = {}
            for node in ("publish-a", "publish-b"):
                packet = study.prefix(case, local, len(records) + 1, continuation=continuation)
                record("prefix", phase="before_publisher", continuation=continuation, before_node=node, packet=packet)
                write(directory / f"PREFIX_{continuation:02d}_{node}.json", packet)
                reply = await call(node, messages, continuation)
                if reply is not None:
                    local[node] = reply
                    labels[node] = study.label(case, reply)
            value = {"index": continuation, "completed": len(labels) == 2, "labels": labels}
            write(directory / f"CONTINUATION_{continuation:02d}.json", value)
            continuations.append(value)
    summary = {"shared_labels": shared, "upstream_complete": upstream_complete, "continuations": continuations}
    write(directory / "OUTCOME.json", summary)
    calls = sum(r["kind"] == "request" for r in records)
    report = {"id": case["id"], "base_task": case["base_task"], "api_calls": calls,
              "completed": upstream_complete and len(continuations) == study.CONTINUATIONS and all(c["completed"] for c in continuations),
              "upstream_complete": upstream_complete, "complete_continuations": sum(c["completed"] for c in continuations),
              "failures": failures, "plan_sha256": sha(out / "PLAN.json"),
              "files_sha256": {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()}}
    write(directory / "REPORT.json", report)
    return report


async def collect(out):
    from paperalpha_runtime.quadsentinel_adapter import create_api_client
    out = Path(out).resolve()
    plan = load_plan(out)
    if (out / "runs").exists() or (out / "COLLECTION.json").exists() or (out / "COLLECT_ENVIRONMENT.json").exists():
        raise ValueError("existing run must not be restarted; inspect its actual status")
    write(out / "COLLECT_ENVIRONMENT.json", runtime_environment("collect"))
    api = create_api_client(plan["api"]["endpoint"], os.environ.get(plan["api"]["key_environment_variable"]))
    limiter = asyncio.Semaphore(plan["api"]["concurrency"])

    async def bounded(case):
        async with limiter:
            report = await collect_group(out, case, api)
            print(json.dumps({k: report[k] for k in ("id", "api_calls", "completed", "complete_continuations")}), flush=True)
            return report

    try:
        reports = await asyncio.gather(*(bounded(case) for case in plan["cases"]))
    finally:
        await api.close()
    report = {"plan_sha256": sha(out / "PLAN.json"), "groups": len(reports),
              "collect_environment_sha256": sha(out / "COLLECT_ENVIRONMENT.json"),
              "api_calls": sum(r["api_calls"] for r in reports), "complete_groups": sum(r["completed"] for r in reports),
              "complete_continuations": sum(r["complete_continuations"] for r in reports),
              "reports_sha256": {r["id"]: sha(out / "runs" / r["id"] / "REPORT.json") for r in reports},
              "development_only": True, "new_fits": 0, "prospective_predictions": 0}
    if report["api_calls"] > plan["maximum_api_calls"]:
        raise ValueError("API budget exceeded")
    write(out / "COLLECTION.json", report)
    return report


def verify_group(out, case):
    """Reconstruct one group from its original request/response journal only."""
    out = Path(out).resolve()
    directory = out / "runs" / case["id"]
    report = read(directory / "REPORT.json")
    if report["id"] != case["id"] or report["plan_sha256"] != sha(out / "PLAN.json"):
        raise ValueError("report identity or frozen plan mismatch")
    actual_files = {p.name for p in directory.iterdir() if p.is_file()} - {"REPORT.json"}
    if actual_files != set(report["files_sha256"]):
        raise ValueError("missing or extra group artifacts")
    for name, expected in report["files_sha256"].items():
        if sha(directory / name) != expected:
            raise ValueError("raw group artifact changed: " + name)
    records = [read(p) for p in sorted(directory.glob("record_*.json"))]
    if [r["sequence"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("journal sequence mismatch")
    if any(a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(records, records[1:])):
        raise ValueError("journal clock went backwards")
    for record in records:
        if record["record_id"] != (directory / f"record_{record['sequence']:04d}.json").as_posix():
            raise ValueError("response evidence lacks exact cohort/group/file scope")
    cursor, calls, failure_ids = 0, 0, []

    def take(kind, **expected):
        nonlocal cursor
        if cursor >= len(records):
            raise ValueError("truncated terminal journal")
        value = records[cursor]
        cursor += 1
        if value["kind"] != kind or any(value.get(k) != v for k, v in expected.items()):
            raise ValueError("journal protocol or actual delivery mismatch")
        return value

    def call(node, messages, continuation):
        nonlocal calls
        req = take("request", node=node, continuation=continuation, request=study.request(case, node, messages))
        calls += 1
        common = {"node": node, "continuation": continuation, "request_id": req["record_id"]}
        if cursor < len(records) and records[cursor]["kind"] == "failure":
            fail = take("failure", failure_kind="transport", **common)
            if not isinstance(fail.get("error_type"), str) or not fail["error_type"]:
                raise ValueError("transport error category required")
            failure_ids.append(fail["record_id"])
            return None
        res = take("response", **common)
        try:
            text = response_text(res["response"])
        except (ValueError, KeyError, TypeError) as exc:
            fail = take("failure", failure_kind="invalid_response", error_type=type(exc).__name__, **common)
            failure_ids.append(fail["record_id"])
            return None
        return {"record_id": res["record_id"], "sequence": res["sequence"], "text": text}

    source = take("source", text=case["source"])
    completed = {"source": {k: source[k] for k in ("record_id", "sequence", "text")}}
    messages = {"source": case["source"]}
    early = study.prefix(case, completed, cursor + 1)
    take("prefix", phase="early", continuation=None, before_node=None, packet=early)
    if read(directory / "EARLY.json") != early:
        raise ValueError("early prefix mismatch")
    shared = {}
    for node in study.reviewers(case):
        reply = call(node, messages, None)
        if reply is None:
            break
        completed[node], messages[node] = reply, reply["text"]
        shared[node] = study.label(case, reply)
    upstream_complete = set(shared) == set(study.reviewers(case))
    if read(directory / "SHARED.json") != {"completed": upstream_complete, "labels": shared}:
        raise ValueError("shared labels differ from actual responses")
    outcomes, continuations = [], []
    if upstream_complete:
        for continuation in range(study.CONTINUATIONS):
            local, labels = dict(completed), {}
            for node in ("publish-a", "publish-b"):
                packet = study.prefix(case, local, cursor + 1, continuation=continuation)
                take("prefix", phase="before_publisher", continuation=continuation, before_node=node, packet=packet)
                if read(directory / f"PREFIX_{continuation:02d}_{node}.json") != packet:
                    raise ValueError("continuation pre-call evidence mismatch")
                study.compile_dag(packet)
                reply = call(node, messages, continuation)
                if reply is not None:
                    local[node] = reply
                    labels[node] = study.label(case, reply)
            value = {"index": continuation, "completed": len(labels) == 2, "labels": labels}
            if read(directory / f"CONTINUATION_{continuation:02d}.json") != value:
                raise ValueError("continuation labels differ from actual responses")
            outcomes.append(value)
            if value["completed"]:
                continuations.append(labels)
    if cursor != len(records):
        raise ValueError("unexpected extra journal records or repeated calls")
    if read(directory / "OUTCOME.json") != {"shared_labels": shared, "upstream_complete": upstream_complete, "continuations": outcomes}:
        raise ValueError("group outcome mismatch")
    complete = upstream_complete and len(continuations) == study.CONTINUATIONS
    if (report["api_calls"] != calls or calls > study.planned_calls(case) or report["completed"] != complete
            or report["upstream_complete"] != upstream_complete or report["complete_continuations"] != len(continuations)
            or report["failures"] != failure_ids):
        raise ValueError("group summary differs from replay")
    group = {"id": case["id"], "base_task": case["base_task"], "layout": case["layout"], "repeat": case["repeat"],
             "dag": study.compile_dag(early), "shared_labels": shared, "continuations": continuations,
             "complete": complete, "upstream_complete": upstream_complete, "api_calls": calls,
             "failures": failure_ids, "incomplete_continuations": [v for v in outcomes if not v["completed"]],
             "completed_response_receipts": [{"record_id": r["record_id"], "provider_response_id": r["response"].get("id"),
                                              "returned_model": r["response"].get("model")}
                                             for r in records if r["kind"] == "response" and r["response"].get("status") == "completed"]}
    audit_response_identity([group], scope="within_group_only")
    return group


def verify(out):
    """Read-only: return DAG objects for grouped training; never fit or write."""
    out = Path(out).resolve()
    plan = load_plan(out)
    collection = read(out / "COLLECTION.json")
    if collection["plan_sha256"] != sha(out / "PLAN.json"):
        raise ValueError("collection plan mismatch")
    check_environment(out / "COLLECT_ENVIRONMENT.json", collection["collect_environment_sha256"], "collect")
    if set(collection["reports_sha256"]) != {c["id"] for c in plan["cases"]}:
        raise ValueError("all planned groups must be accounted for")
    for identity, expected in collection["reports_sha256"].items():
        if sha(out / "runs" / identity / "REPORT.json") != expected:
            raise ValueError("group report changed")
    groups = [verify_group(out, case) for case in plan["cases"]]
    actual = {"groups": len(groups), "api_calls": sum(g["api_calls"] for g in groups),
              "complete_groups": sum(g["complete"] for g in groups),
              "complete_continuations": sum(len(g["continuations"]) for g in groups)}
    if any(collection[k] != v for k, v in actual.items()) or actual["api_calls"] > plan["maximum_api_calls"]:
        raise ValueError("collection summary mismatch")
    identity_audit = audit_response_identity(groups, scope="complete_collection")
    return {"schema": "verified-repeated-message-development-v1", **actual, "grouped_traces": groups,
            "eligible_groups": sum(g["eligible_for_grouped_fit"] for g in groups),
            "response_identity_audit": identity_audit,
            "selection_warning": "Fitting only complete publisher pairs with qualified provider response identities conditions on successful completion and identity qualification; preserve failed, partial and excluded groups in coverage reports.",
            "development_only": True, "prospective_predictions": 0}


def serializable_verified(value):
    return {**value, "grouped_traces": [{**g, "dag": g["dag"].specification()} for g in value["grouped_traces"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "collect", "verify"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode == "prepare":
        plan = prepare(out)
        result = {"prepared": str(out), "maximum_api_calls": plan["maximum_api_calls"], "plan_sha256": sha(out / "PLAN.json")}
    else:
        if ROOT != out / "source":
            raise ValueError("run the unchanged script from this output's frozen source")
        result = asyncio.run(collect(out)) if args.mode == "collect" else serializable_verified(verify(out))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
