"""Bounded, cached initial-request/schema extraction. No future text sent."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.canonical import digest
from paperalpha_runtime.task_field_semantics import context_from_initial, request_payload, validate_annotation, unknown_annotation, PROMPT, SCHEMA, SHAPES
from paperalpha_runtime.semantic_evidence import response_annotation


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(parent, out):
    plan = read(parent / "PLAN.json")
    for name, expected in plan["inputs_sha256"].items():
        if sha(parent / "inputs" / name) != expected: raise ValueError("original development inventory changed")
    contexts, bindings = {}, []
    for entry in read(parent / "inputs/provenance.json"):
        if entry["split"] not in ("train", "validation"): raise ValueError("old test forbidden")
        path = Path(plan["upstream"]) / entry["source_path"]
        if sha(path) != entry["source_sha256"]: raise ValueError("raw initial source changed")
        initial = read(path)["input"]
        context, binding = context_from_initial(initial["request"], initial["vault"])
        identity = digest(context)
        if identity in contexts and context != contexts[identity]: raise ValueError("context collision")
        contexts[identity] = context
        bindings.append({"id": entry["id"], "source_path": entry["source_path"], "source_sha256": entry["source_sha256"], **binding})
    if len(contexts) > 64: raise ValueError("context count exceeds frozen call budget; no partial selective extraction")
    out.mkdir(parents=True, exist_ok=False)
    names = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'code/src').rglob("*.py")]
    names += ['code/commands/collect_task_field_semantics.py']
    for name in names:
        dest = out / "source" / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dest)
    for identity, context in contexts.items(): write(out / "contexts" / (identity + ".json"), context)
    write(out / "SOURCE_BINDINGS.json", bindings)
    write(out / "PLAN.json", {"schema": "initial-task-field-extraction-v1", "parent": str(parent),
        "parent_plan_sha256": sha(parent / "PLAN.json"), "upstream": plan["upstream"], "created_at_utc": now(),
        "source_sha256": {name: sha(out / "source" / name) for name in names},
        "contexts_sha256": {k + ".json": sha(out / "contexts" / (k + ".json")) for k in contexts},
        "bindings_sha256": sha(out / "SOURCE_BINDINGS.json"), "prompt": PROMPT, "schema_definition": SCHEMA,
        "normalization_shapes": SHAPES, "model": "gpt-5.5", "reasoning_effort": "low",
        "max_calls": len(contexts), "attempts_per_context": 1, "timeout_seconds": 120,
        "fallback": "All unclear, retained for every failed context; never drop flows or silently repair categories.",
        "input": "Normalized initial task + complete schema names/types only; no values, policy, labels, template/domain/model identities or later messages.",
        "scope": "Previously seen development inputs. Offline initial-only features, not new agent behavior or historical online extraction. Subject identity matching is outside scope."})
    print(json.dumps({"prepared": str(out), "flows": len(bindings), "contexts": len(contexts),
        "masked": sum(b["masked_span"] is not None for b in bindings),
        "tasks": sorted({c["request_text"] for c in contexts.values()})}), flush=True)


async def collect(out):
    from openai import AsyncOpenAI
    from urllib.parse import urlsplit
    plan = read(out / "PLAN.json")
    if ROOT != out / "source": raise ValueError("execute frozen collector")
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected: raise ValueError("frozen source changed")
    for name, expected in plan["contexts_sha256"].items():
        if sha(out / "contexts" / name) != expected: raise ValueError("frozen context changed")
    if sha(out / "SOURCE_BINDINGS.json") != plan["bindings_sha256"]: raise ValueError("source bindings changed")
    base = os.environ.get("PAPERALPHA_API_BASE", "http://localhost:58661/v1")
    url = urlsplit(base)
    if url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1") or url.port != 58661 or url.path != "/v1" or url.query or url.fragment or url.username:
        raise ValueError("only user-authorized local endpoint accepted")
    key = os.environ.get("PAPERALPHA_API_KEY")
    if not key: raise ValueError("PAPERALPHA_API_KEY required")
    write(out / "START.json", {"started_at_utc": now(), "plan_sha256": sha(out / "PLAN.json")})
    semaphore = asyncio.Semaphore(2)
    async with AsyncOpenAI(base_url=base, api_key=key, max_retries=0, timeout=plan["timeout_seconds"]) as client:
        async def one(name):
            async with semaphore:
                context = read(out / "contexts" / name); payload = request_payload(context)
                write(out / "requests" / name, payload)
                result = {"context_sha256": digest(context), "started_at_utc": now()}
                try:
                    raw = (await client.responses.create(**payload)).model_dump(mode="json")
                    write(out / "responses" / name, raw)
                    result["usage"] = raw.get("usage")
                    try:
                        result["annotation"] = validate_annotation(context, response_annotation(raw))
                        result["status"] = "quote_anchored_hypothesis"
                    except (ValueError, TypeError, KeyError):
                        result.update(status="invalid_annotation", annotation=unknown_annotation(context))
                except Exception as exc:
                    # Never persist client exception text/headers containing credentials.
                    result.update(status="api_error", error_type=type(exc).__name__, annotation=unknown_annotation(context))
                result["completed_at_utc"] = now()
                write(out / "classifications" / name, result)
                print(json.dumps({"context": name, "status": result["status"]}), flush=True)
                return result
        results = await asyncio.gather(*(one(name) for name in sorted(plan["contexts_sha256"])))
    report = {"completed_at_utc": now(), "plan_sha256": sha(out / "PLAN.json"), "api_calls": len(results),
        "outcomes": dict(Counter(r["status"] for r in results)),
        "input_tokens": sum((r.get("usage") or {}).get("input_tokens", 0) for r in results),
        "output_tokens": sum((r.get("usage") or {}).get("output_tokens", 0) for r in results),
        "internal_categories": dict(Counter(f["internal_use"] for r in results for f in r["annotation"]["fields"])),
        "outward_categories": dict(Counter(f["outward_form"] for r in results for f in r["annotation"]["fields"])),
        "semantic_truth_verified": False, "new_target_agent_calls": 0,
        "artifact_sha256": {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    write(out / "REPORT.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "artifact_sha256"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("prepare", "collect"))
    parser.add_argument("--parent", type=Path); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args.parent.resolve(), args.out.resolve())
    else: asyncio.run(collect(args.out.resolve()))
