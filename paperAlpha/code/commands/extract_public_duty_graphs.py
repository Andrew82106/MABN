"""Four bounded API extractions: two public tasks, original and opaque actor IDs."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys

from harnessaudit_cohort import ROOT, sha, write_new, now
from collect_harnessaudit_api_pilot import validate_endpoint

sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.duty_graph import public_packet, request_payload, validate_duties, duty_vectors, PROMPT, SCHEMA
from paperalpha_runtime.semantic_evidence import response_annotation, digest


async def extract(output):
    from openai import AsyncOpenAI
    if output.exists():
        raise ValueError("new output directory required")
    endpoint = validate_endpoint(os.environ.get("PAPERALPHA_API_BASE", "http://localhost:58661/v1"))
    key = os.environ.get("PAPERALPHA_API_KEY")
    if not key:
        raise ValueError("PAPERALPHA_API_KEY required")
    output.mkdir(parents=True)
    packets = []
    for task in ("dl-t20", "dl-t22"):
        run = ROOT / 'experiments/model_mechanisms/runs/stage_dag_development_20260919/runs' / task
        manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
        path = run / "runtime_events.jsonl"
        if sha(path) != manifest["artifact_sha256"][path.name]:
            raise ValueError("original public runtime setup changed")
        with path.open(encoding="utf-8") as handle:
            first = json.loads(next(handle))
        if first["kind"] != "run_started" or first["sequence"] != 1:
            raise ValueError("original pre-run setup required")
        config = first["payload"]["public_configuration"]
        for anonymous in (False, True):
            packet, aliases = public_packet(config, anonymize=anonymous)
            identity = task + ("__opaque" if anonymous else "__original")
            item = {"id": identity, "task_id": task, "opaque_ids": anonymous, "packet": packet, "aliases": aliases,
                    "configuration_sha256": digest(config), "source_journal_sha256": sha(path)}
            write_new(output / "inputs" / (identity + ".json"), item)
            packets.append(item)
    sources = [Path(__file__), ROOT / 'code/src/paperalpha_runtime/duty_graph.py',
        ROOT / 'code/src/paperalpha_runtime/semantic_evidence.py', ROOT / 'code/src/paperalpha_runtime/initial_task_context.py',
        ROOT / 'code/commands/harnessaudit_cohort.py', ROOT / 'code/commands/collect_harnessaudit_api_pilot.py']
    hashes = {}
    for path in sources:
        target = output / "source_snapshot" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(path.read_bytes())
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    write_new(output / "SPEC.json", {"created_at": now(), "max_calls": 4, "attempts": 1, "model": "gpt-5.5",
        "reasoning": "low", "max_output_tokens": 6500, "prompt": PROMPT, "schema": SCHEMA,
        "source_sha256": hashes, "semantic_results_reviewed_before_fitting": True,
        "scope": "Previously used development tasks, no labels/runtime text/hidden policies supplied; opaque-ID calls test extraction sensitivity only."})
    gate = asyncio.Semaphore(2)
    async with AsyncOpenAI(api_key=key, base_url=endpoint, max_retries=0, timeout=150) as client:
        async def one(item):
            async with gate:
                identity = item["id"]
                payload = request_payload(item["packet"])
                write_new(output / "requests" / (identity + ".json"), payload)
                print("extract public duties", identity, flush=True)
                result = {"id": identity, "packet_sha256": item["packet"]["sha256"]}
                try:
                    response = await client.responses.create(**payload)
                    raw = response.model_dump(mode="json")
                    write_new(output / "responses" / (identity + ".json"), raw)
                    result["usage"] = raw.get("usage")
                    try:
                        annotation = response_annotation(raw)
                        vectors, graph = duty_vectors(item["packet"], annotation, aliases=item["aliases"])
                        result.update(status="quote_anchored", annotation=annotation, graph=graph, vectors=vectors)
                    except (ValueError, TypeError, KeyError) as exc:
                        result.update(status="invalid_output", reason=str(exc))
                except Exception as exc:
                    result.update(status="api_error", error_type=type(exc).__name__)
                result["available_at"] = now()
                write_new(output / "extractions" / (identity + ".json"), result)
                print(identity, result["status"], flush=True)
                return result
        results = await asyncio.gather(*(one(item) for item in packets))
    report = {"completed_at": now(), "api_calls": len(results), "outcomes": dict(Counter(r["status"] for r in results)),
        "input_tokens": sum((r.get("usage") or {}).get("input_tokens", 0) for r in results),
        "output_tokens": sum((r.get("usage") or {}).get("output_tokens", 0) for r in results),
        "training": False, "live_workflow_execution": False,
        "artifacts_sha256": {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}}
    write_new(output / "REPORT.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "artifacts_sha256"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(extract(args.out.resolve()))
