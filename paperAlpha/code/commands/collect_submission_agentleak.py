"""Prepare/collect the single 16-template submission development cohort.

prepare and verify do not contact an API. Real collection is an explicit run
command and uses PAPERALPHA_LOCAL_API_KEY only. No estimator or old test loads.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime import agentleak_submission as study


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def durable_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def replace_summary(path, value):
    """Derived summaries may be refreshed; immutable API receipts never are."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def collection_lock(out):
    """OS lock releases after a crash; a stale filename does not block resume."""
    with (Path(out) / "COLLECTION.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare(out=study.DEFAULT_OUTPUT, seed=study.DEFAULT_SEED, upstream=study.DEFAULT_UPSTREAM):
    out, upstream = Path(out).resolve(), Path(upstream).resolve()
    if out.exists():
        raise ValueError("new cohort directory required; existing frozen generation is never replaced")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer generation seed required")
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--",
                                     *study.UPSTREAM_SOURCES], text=True).strip()
    if revision != study.EXPECTED_UPSTREAM_COMMIT or dirty:
        raise ValueError("reviewed official commit and unchanged generation/reference sources required")
    cases = study.generate_cases(upstream, seed)
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for prefix, base, names in (("local", ROOT, study.LOCAL_SOURCES),
                                ("upstream", upstream, study.UPSTREAM_SOURCES)):
        for name in names:
            relative = prefix + "/" + name
            target = out / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(base / name, target)
            sources[relative] = study.sha(target)
    packages = {}
    for name in ("pydantic", "Faker", "tzdata", "openai", "httpx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    plan = {"schema": "submission-agentleak-development-v1", "created_at": study.now(),
            "seed": seed, "api": study.API, "maximum_api_calls": 48, "cases": cases,
            "source_sha256": sources, "upstream": str(upstream), "upstream_commit": revision, "protocol": study.protocol(),
            "environment": {"python": sys.version, "packages": packages},
            "agent_graph": {"nodes": list(study.STAGES), "edges": [["coordinator", "worker"], ["worker", "finalizer"]]},
            "information_routes": "Frozen source records enter coordinator and worker; finalizer receives worker text and task only"}
    plan["sha256"] = study.digest(plan)
    durable_new(out / "PLAN.json", plan)
    return plan


def load_plan(out):
    out = Path(out)
    plan = read(out / "PLAN.json")
    if plan.get("schema") != "submission-agentleak-development-v1" or study.digest(
            {k: v for k, v in plan.items() if k != "sha256"}) != plan.get("sha256"):
        raise ValueError("frozen plan changed")
    if plan["api"] != study.API or plan["maximum_api_calls"] != 48 or len(plan["cases"]) != 16:
        raise ValueError("fixed sixteen-case forty-eight-call protocol required")
    if plan.get("upstream_commit") != study.EXPECTED_UPSTREAM_COMMIT:
        raise ValueError("reviewed upstream commit required")
    expected_ids = {f"{v}-{i:02d}" for v in ("healthcare", "finance", "legal", "corporate") for i in range(1, 5)}
    if {case["case_id"] for case in plan["cases"]} != expected_ids:
        raise ValueError("all task templates required exactly once")
    for case in plan["cases"]:
        if len(case["records"]) != 3 or study.field_registry(case["records"]) != case["field_registry"]:
            raise ValueError("complete multi-object registry required")
    for name, expected in plan["source_sha256"].items():
        if study.sha(out / "source" / name) != expected:
            raise ValueError("frozen source changed: " + name)
        if name.startswith("local/") and study.sha(ROOT / name.removeprefix("local/")) != expected:
            raise ValueError("current collector differs from frozen source: " + name)
    return plan


def stage_folder(out, case, index):
    return Path(out) / "cases" / case["case_id"] / "stages" / f"{index + 1:02d}-{study.STAGES[index]}"


def case_trace(out, case, plan):
    """Reconstruct only from persisted receipts, not generated expectations."""
    messages, stages, status = [], [], "not_started"
    for index, stage in enumerate(study.STAGES):
        folder = stage_folder(out, case, index)
        attempt_path = folder / "attempt.json"
        if not attempt_path.exists():
            break
        attempt = read(attempt_path)
        expected = study.stage_request(case, index, messages)
        before = study.prefix(case, messages)
        if (attempt.get("plan_sha256") != plan["sha256"] or attempt.get("request") != expected
                or attempt.get("prefix_before") != before or attempt.get("prefix_sha256") != study.digest(before)):
            raise ValueError("attempt differs from frozen task or actual prefix")
        row = {"stage": stage, "request_seq": attempt["request_seq"], "request": expected, "prefix_before": before,
               "status": "attempted_unknown", "response": None, "delivered_text": None,
               "prefix_after": before}
        if (folder / "failure.json").exists():
            row["status"] = "failed"
            row["failure"] = read(folder / "failure.json")
        if (folder / "response.json").exists():
            response = read(folder / "response.json")
            if response.get("request_sha256") != study.digest(expected):
                raise ValueError("response/request binding changed")
            row["response"] = response["response"]
            try:
                visible = study.visible_response(row["response"])
                if "failure" in row:
                    # A persisted failure is never silently overwritten by a
                    # valid response. This is not a normal collection path,
                    # but may reflect interrupted local post-processing.
                    row["status"] = "failed_after_response"
                else:
                    messages.append(visible["text"])
                    row.update(status="completed", delivered_text=visible["text"], native_refusal=visible["native_refusal"],
                               prefix_after=study.prefix(case, messages))
            except ValueError:
                row["status"] = "invalid_response"
        stages.append(row)
        status = row["status"]
        if status != "completed":
            break
    complete = len(messages) == 3
    channel_rows = study.channels(messages)
    labels = {"complete": complete,
              "stage_field_labels": [{"stage": study.STAGES[i], **study.literal_labels(case, text)} for i, text in enumerate(messages)],
              "channel_labels": [{"channel": item["channel"], "source_stage": item["source_stage"],
                                  **study.literal_labels(case, item["content"])} for item in channel_rows],
              "final_literal_policy_violation": study.literal_labels(case, messages[-1])["y"] if complete else None}
    return {"schema": "submission-agentleak-trace-v1", "case_id": case["case_id"], "template_id": case["template_id"],
            "plan_sha256": plan["sha256"], "status": "completed" if complete else ("partial" if status == "completed" else status),
            "stages": stages, "channels": channel_rows, "labels": labels}


async def collect_case(out, case, plan, transport, request_seq):
    trace = case_trace(out, case, plan)
    if trace["status"] not in {"not_started", "partial"}:
        return trace
    messages = [s["delivered_text"] for s in trace["stages"]]
    for index in range(len(messages), 3):
        folder = stage_folder(out, case, index)
        request = study.stage_request(case, index, messages)
        before = study.prefix(case, messages)
        request_seq[0] += 1
        if request_seq[0] > plan["maximum_api_calls"]:
            raise ValueError("frozen call budget exhausted")
        durable_new(folder / "attempt.json", {"case_id": case["case_id"], "stage": study.STAGES[index],
            "request_seq": request_seq[0], "at": study.now(), "plan_sha256": plan["sha256"],
            "prefix_before": before, "prefix_sha256": study.digest(before), "request": request})
        try:
            raw = await asyncio.wait_for(transport(request), timeout=plan["api"]["timeout_seconds"])
            response = study.canonical_response(raw)
            durable_new(folder / "response.json", {"received_at": study.now(),
                "request_sha256": study.digest(request), "response": response})
            visible = study.visible_response(response)
            messages.append(visible["text"])
        except Exception as exc:
            # Exception messages/HTTP bodies may contain credentials: never save them.
            durable_new(folder / "failure.json", {"at": study.now(), "error_type": type(exc).__name__,
                "http_status": getattr(exc, "status_code", None), "retry_allowed": False})
            break
    trace = case_trace(out, case, plan)
    replace_summary(Path(out) / "cases" / case["case_id"] / "TRACE.json", trace)
    return trace


def verify(out, plan=None):
    out = Path(out)
    plan = plan or load_plan(out)
    traces = [case_trace(out, case, plan) for case in plan["cases"]]
    attempts = [stage for trace in traces for stage in trace["stages"]]
    if len(attempts) > 48 or len({s["request_seq"] for s in attempts}) != len(attempts):
        raise ValueError("request count or sequence identity invalid")
    response_ids = {}
    for trace in traces:
        for stage in trace["stages"]:
            if stage["response"] and stage["response"].get("id"):
                response_ids.setdefault(stage["response"]["id"], []).append(trace["case_id"])
    excluded = sorted({case for cases in response_ids.values() if len(cases) > 1 for case in cases})
    return {"schema": "submission-agentleak-collection-v1", "plan_sha256": plan["sha256"],
        "cases": [{"case_id": trace["case_id"], "template_id": trace["template_id"], "status": trace["status"],
                   "attempted_requests": len(trace["stages"]), "final_label": trace["labels"]["final_literal_policy_violation"],
                   "identity_eligible": trace["case_id"] not in excluded} for trace in traces],
        "api_attempts": len(attempts), "maximum_api_calls": 48,
        "completed_cases": sum(trace["status"] == "completed" for trace in traces),
        "response_identity_exclusions": excluded, "fits": 0, "risk_scores": None,
        "limits": "Development only; incomplete/failed cases are not safe; duplicated IDs excluded from future analysis; fields/calls are not independent tasks"}


async def collect(out, transport):
    """Injected transport supports fake testing; only CLI run creates an API client."""
    out = Path(out)
    plan = load_plan(out)
    with collection_lock(out):
        previous = verify(out, plan)
        sequence = [previous["api_attempts"]]
        semaphore = asyncio.Semaphore(plan["api"]["concurrency"])

        async def one(case):
            async with semaphore:
                trace = await collect_case(out, case, plan, transport, sequence)
                # Reconstruct summaries after a crash without any new API request.
                folder = out / "cases" / case["case_id"]
                folder.mkdir(parents=True, exist_ok=True)
                replace_summary(folder / "TRACE.json", trace)
                print(json.dumps({"case": case["case_id"], "status": trace["status"],
                                  "attempts": len(trace["stages"])}), flush=True)

        await asyncio.gather(*(one(case) for case in plan["cases"]))
        result = verify(out, plan)
        replace_summary(out / "COLLECTION.json", result)
        return result


def validate_endpoint(endpoint):
    parsed = urlparse(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"}
            or parsed.port != 58661 or parsed.path.rstrip("/") != "/v1"
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("explicit authorized local endpoint http://localhost:58661/v1 required")
    return endpoint.rstrip("/")


async def run_api(out, endpoint):
    endpoint = validate_endpoint(endpoint)
    load_plan(out)
    key = os.environ.get("PAPERALPHA_LOCAL_API_KEY")
    if not key:
        raise ValueError("process-local PAPERALPHA_LOCAL_API_KEY required")
    from openai import AsyncOpenAI
    async with AsyncOpenAI(base_url=endpoint, api_key=key, max_retries=0,
                           timeout=study.API["timeout_seconds"]) as client:
        async def transport(request):
            return await client.responses.create(**request)
        return await collect(out, transport)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=study.DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=study.DEFAULT_SEED)
    parser.add_argument("--endpoint", default="http://localhost:58661/v1")
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.out, args.seed)
        print(json.dumps({"prepared": str(args.out.resolve()), "tasks": len(result["cases"]),
                          "maximum_api_calls": result["maximum_api_calls"], "plan_sha256": result["sha256"]}))
    elif args.mode == "verify":
        print(json.dumps(verify(args.out), ensure_ascii=False))
    else:
        print(json.dumps(asyncio.run(run_api(args.out, args.endpoint)), ensure_ascii=False))
