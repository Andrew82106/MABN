"""Four bounded API classifications of initial, hub-visible MAS duties.

Uses only each frozen run's first request and run-start record. Later text,
labels, task YAML and hidden policies are not sent to the classifier.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from harnessaudit_cohort import ROOT, load_plan, sha, write_new
from collect_harnessaudit_api_pilot import validate_endpoint

sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.initial_task_context import from_initial_request, request_payload, validate_roles, PROMPT, SCHEMA
from paperalpha_runtime.semantic_evidence import response_annotation, digest


def now():
    return datetime.now(timezone.utc).isoformat()


def first_record(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError("empty source")


def contexts_from_plan(plan_path):
    plan = load_plan(plan_path)
    if len(plan["tasks"]) != 4:
        raise ValueError("this bounded probe expects the four-task development cohort")
    contexts = []
    for task in plan["tasks"]:
        run = plan_path.parent / "runs" / task["task_id"]
        manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
        for name in ("api_requests.jsonl", "runtime_events.jsonl"):
            if sha(run / name) != manifest["artifact_sha256"][name]:
                raise ValueError("original acquisition input changed")
        request, start = first_record(run / "api_requests.jsonl"), first_record(run / "runtime_events.jsonl")
        contexts.append({"task_id": task["task_id"], "initial_request": request, "run_start": start,
                         "context": from_initial_request(request, start)})
    return plan, contexts


async def extract(plan_path, output):
    from openai import AsyncOpenAI
    plan_path, output = plan_path.resolve(), output.resolve()
    if output.exists():
        raise ValueError("new extraction directory required")
    endpoint = validate_endpoint(os.environ.get("PAPERALPHA_API_BASE", "http://localhost:58661/v1"))
    key = os.environ.get("PAPERALPHA_API_KEY")
    if not key:
        raise ValueError("PAPERALPHA_API_KEY required")
    plan, contexts = contexts_from_plan(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / 'code/commands/harnessaudit_cohort.py', ROOT / 'code/commands/collect_harnessaudit_api_pilot.py',
               *(ROOT / 'code/src/paperalpha_runtime').glob("*.py")]
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        dest = output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as handle:
            handle.write(source.read_bytes())
        hashes[relative.as_posix()] = sha(dest)
    for item in contexts:
        write_new(output / "inputs" / f"{item['task_id']}.json", item)
    write_new(output / "SPEC.json", {"prompt": PROMPT, "schema": SCHEMA, "max_calls": 4, "attempts_per_task": 1,
                                    "model": "gpt-5.5", "api_reasoning_effort": "low", "max_output_tokens": 1600,
                                    "source_sha256": hashes, "plan_sha256": plan["plan_sha256"],
                                    "started_at": now(), "offline": True})
    gate = asyncio.Semaphore(2)
    async with AsyncOpenAI(api_key=key, base_url=endpoint, max_retries=0, timeout=60) as client:
        async def one(item):
            async with gate:
                identity, context = item["task_id"], item["context"]
                payload = request_payload(context)
                write_new(output / "requests" / f"{identity}.json", payload)
                print(f"classify initial duties: {identity}", flush=True)
                result = {"task_id": identity, "context_sha256": context["context_sha256"]}
                try:
                    response = await client.responses.create(**payload)
                    raw = response.model_dump(mode="json")
                    write_new(output / "responses" / f"{identity}.json", raw)
                    result["usage"] = raw.get("usage")
                    try:
                        annotation = response_annotation(raw)
                        validated = validate_roles(context, annotation)
                        result.update(status="quote_anchored", annotation=annotation, evidence=validated)
                    except (ValueError, KeyError, TypeError) as exc:
                        result.update(status="invalid_classification", reason=str(exc))
                except Exception as exc:
                    result.update(status="api_error", error_type=type(exc).__name__)
                result["classification_available_at"] = now()
                write_new(output / "classifications" / f"{identity}.json", result)
                return result
        results = await asyncio.gather(*(one(item) for item in contexts))
    report = {"finished_at": now(), "plan_sha256": plan["plan_sha256"], "api_calls": len(results),
              "outcomes": dict(Counter(r["status"] for r in results)),
              "input_tokens": sum((r.get("usage") or {}).get("input_tokens", 0) for r in results),
              "output_tokens": sum((r.get("usage") or {}).get("output_tokens", 0) for r in results),
              "role_counts": dict(Counter(role for r in results for role in r.get("evidence", {}).get("roles", {}).values())),
              "scope": "Offline role-classifier probe; exact quotes do not establish correct categories, permissions, DAGs or model performance.",
              "artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write_new(output / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(extract(args.plan, args.out))
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_sha256"}, ensure_ascii=False))
