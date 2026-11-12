"""Bounded offline semantic extraction from journal prefixes, no outcome input."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.semantic_evidence import digest
from collect_harnessaudit_api_pilot import validate_endpoint


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def extract(args):
    from openai import AsyncOpenAI
    protocol_name = getattr(args, "protocol", "v1")
    if protocol_name not in {"v1", "v2"}:
        raise ValueError("explicit supported semantic protocol required")
    module = importlib.import_module("paperalpha_runtime.semantic_evidence" + ("_v2" if protocol_name == "v2" else ""))
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError("extraction output must be new")
    endpoint = validate_endpoint(os.environ.get("PAPERALPHA_API_BASE", "http://localhost:58661/v1"))
    key = os.environ.get("PAPERALPHA_API_KEY")
    if not key:
        raise ValueError("PAPERALPHA_API_KEY is required; no credential file is read")
    raw = args.input.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    window = module.TextEvidenceWindow()
    # Each packet depends only on observations through its own sequence. No
    # extraction response is fed into another request, so concurrency cannot
    # leak future claims or compound an earlier model annotation as fact.
    packets = [packet for row in records if (packet := window.push(row)) is not None]
    chosen = packets[:args.limit]
    if not chosen:
        raise ValueError("no communication/handler-return text in this journal")
    output.mkdir(parents=True)
    (output / "runtime_events.jsonl").write_bytes(raw)
    snapshots = {}
    for path in [Path(__file__), ROOT / 'code/commands/collect_harnessaudit_api_pilot.py', *(ROOT / 'code/src/paperalpha_runtime').glob("*.py")]:
        relative = path.resolve().relative_to(ROOT.resolve())
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        snapshots[relative.as_posix()] = sha(destination)
    for packet in chosen:
        append(output / "input_packets.jsonl", packet)
    save(output / "extraction_specification.json", {"prompt": module.PROMPT, "schema": module.OUTPUT_SCHEMA,
        "window": {"max_prior_records": 6, "max_context_characters": 16000},
        "label_policy": "no outcomes supplied; no extracted observation is a safety or authorization label"})
    manifest = {"status": "running", "semantic_protocol": protocol_name, "model_requested": "gpt-5.5", "endpoint": endpoint,
        "started_at": datetime.now(timezone.utc).isoformat(), "input_sha256": sha(output / "runtime_events.jsonl"),
        "selection": "first text records in runtime order, without outcomes", "available_text_records": len(packets),
        "selected_text_records": len(chosen), "unselected_record_ids": [p["current"]["record_id"] for p in packets[args.limit:]],
        "max_api_calls": args.limit, "concurrency": 2, "client_retries": 0,
        "max_output_tokens": 3200 if protocol_name == "v2" else 2200,
        "api_reasoning_effort": "low", "source_snapshot": snapshots, "openai_version": importlib.metadata.version("openai"),
        "journal_report": window.projector.report(),
        "scope": "Offline text-evidence extraction prototype. No new MAS execution, trained BN, risk accuracy, or verified provenance."}
    save(output / "RUN.json", manifest)
    semaphore = asyncio.Semaphore(2)
    async with AsyncOpenAI(api_key=key, base_url=endpoint, max_retries=0, timeout=60.) as client:
        async def one(index, packet):
            async with semaphore:
                identity = packet["current"]["record_id"]
                started = time.monotonic()
                result = {"call": index, "record_id": identity, "input_packet_sha256": digest(packet)}
                try:
                    request = module.request_payload(packet)
                except ValueError as exc:
                    result.update(status="not_requested", error=str(exc))
                    append(output / "extractions.jsonl", result)
                    return result
                append(output / "api_requests.jsonl", {**result, "request": request})
                print(f"extract {index}/{len(chosen)} {identity}", flush=True)
                try:
                    response = await client.responses.create(**request)
                    payload = response.model_dump(mode="json")
                    append(output / "api_responses.jsonl", {**result, "response": payload, "elapsed_seconds": time.monotonic() - started})
                    result["usage"] = payload.get("usage")
                    try:
                        annotation = module.response_annotation(payload)
                        result.update(status="validated_observation", evidence=module.validate_annotation(packet, annotation))
                    except (ValueError, TypeError, KeyError) as exc:
                        result.update(status="invalid_observation", error=str(exc))
                except Exception as exc:
                    # Do not save arbitrary exception text or headers (keys).
                    result.update(status="api_error", error_type=type(exc).__name__)
                    append(output / "api_responses.jsonl", {**result, "elapsed_seconds": time.monotonic() - started})
                append(output / "extractions.jsonl", result)
                return result
        outcomes = await asyncio.gather(*(one(i, p) for i, p in enumerate(chosen, 1)))
    counts = dict(Counter(r["status"] for r in outcomes))
    manifest.update(status="completed" if set(counts) == {"validated_observation"} else "completed_with_gaps",
        finished_at=datetime.now(timezone.utc).isoformat(), outcome_counts=counts,
        input_tokens=sum((r.get("usage") or {}).get("input_tokens", 0) for r in outcomes),
        output_tokens=sum((r.get("usage") or {}).get("output_tokens", 0) for r in outcomes),
        validated_observations=sum(len(r.get("evidence", {}).get("observations", [])) for r in outcomes),
        validated_reference_anchors=sum(len(o["references"]) for r in outcomes for o in r.get("evidence", {}).get("observations", [])),
        qualification="Validated means schema/quote/reference anchoring only; classifications and semantic links are not independently adjudicated.")
    manifest["artifact_sha256"] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "RUN.json"}
    save(output / "RUN.json", manifest)
    return {k: manifest[k] for k in ("status", "selected_text_records", "outcome_counts", "validated_observations", "validated_reference_anchors", "input_tokens", "output_tokens")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--protocol", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    if not 1 <= args.limit <= 32:
        parser.error("request limit must be 1..32")
    print(json.dumps(asyncio.run(extract(args)), ensure_ascii=False))
