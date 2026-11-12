"""Initial-schema semantic hypotheses; prepare/verify are strictly API-free.

Only the parent PLAN's initial requests and complete record schemas are used.
No TRACE, target response, evaluation label or previous study is ever opened.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.canonical import digest
from paperalpha_runtime.task_field_semantics import (
    context_from_initial, request_payload, validate_annotation, unknown_annotation,
    PROMPT, SCHEMA, SHAPES,
)
from paperalpha_runtime.semantic_evidence import response_annotation

DEFAULT_PARENT = ROOT / 'experiments/privacy_flow/runs/submission__agentleak/PLAN.json'
DEFAULT_OUTPUT = DEFAULT_PARENT.parent / "semantics"
API = {"model": "gpt-5.5", "reasoning": {"effort": "low"}, "max_output_tokens": 6000,
       "store": False, "concurrency": 2, "timeout_seconds": 90, "max_retries": 0,
       "maximum_api_calls": 16, "attempts_per_context": 1}
SOURCES = ('code/commands/collect_submission_semantics.py', 'code/tests/test_submission_semantics.py',
           'code/src/paperalpha_runtime/__init__.py', 'code/src/paperalpha_runtime/canonical.py',
           'code/src/paperalpha_runtime/task_field_semantics.py',
           'code/src/paperalpha_runtime/semantic_evidence.py', 'code/src/paperalpha_runtime/journal.py')


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def durable_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def replace_summary(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def collection_lock(out):
    """Process lock; claims remain permanent, but a crash releases this lock."""
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


def initial_contexts(parent_path):
    """Read one PLAN, never any path referenced by its cases or metadata."""
    parent = read(parent_path)
    if (parent.get("schema") != "submission-agentleak-development-v1"
            or digest({k: v for k, v in parent.items() if k != "sha256"}) != parent.get("sha256")):
        raise ValueError("intact parent AgentLeak PLAN required")
    cases = parent.get("cases", [])
    if len(cases) != 16 or len({c["case_id"] for c in cases}) != 16:
        raise ValueError("all sixteen initial cases required")
    contexts, bindings = {}, []
    for case in cases:
        records = case["records"]
        if len(records) != 3 or len({r["record_id"] for r in records}) != 3:
            raise ValueError("three distinct complete source records required")
        generated = [context_from_initial(case["request"], r["fields"])[0] for r in records]
        if any(item != generated[0] for item in generated):
            raise ValueError("heterogeneous record schemas need a separately reviewed adapter")
        context = generated[0]
        identity = digest(context)
        contexts[identity] = context
        bindings.append({"case_id": case["case_id"], "context_sha256": identity,
            "initial_sha256": digest({"request": case["request"], "records": records})})
    if len(contexts) != 16:
        raise ValueError("exactly sixteen distinct complete initial contexts required")
    return parent, contexts, bindings


def prepare(out=DEFAULT_OUTPUT, parent_path=DEFAULT_PARENT):
    out, parent_path = Path(out).resolve(), Path(parent_path).resolve()
    if out.exists():
        raise ValueError("new output directory required; never overwrite a frozen PLAN")
    parent, contexts, bindings = initial_contexts(parent_path)
    out.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in SOURCES:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        sources[name] = sha(target)
    context_hashes = {}
    for identity, context in sorted(contexts.items()):
        target = out / "contexts" / (identity + ".json")
        durable_new(target, context)
        context_hashes[identity] = sha(target)
    packages = {}
    for name in ("openai", "httpx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    plan = {"schema": "submission-initial-semantics-v1", "created_at": now(),
        "parent_plan_path": str(parent_path), "parent_plan_file_sha256": sha(parent_path),
        "parent_plan_sha256": parent["sha256"], "api": copy.deepcopy(API),
        "bindings": bindings, "contexts_sha256": context_hashes, "source_sha256": sources,
        "prompt": PROMPT, "schema_definition": copy.deepcopy(SCHEMA),
        "normalization_shapes": [list(item) for item in SHAPES],
        "environment": {"python": sys.version, "packages": packages},
        "protocol": {
            "input": "Normalized initial task plus complete field names/types; values, policies, case IDs and future messages never enter the API request.",
            "timing": "Offline extraction from initial-only input; not available during the previously collected agent run.",
            "scope": "No old cache, raw trajectory, TRACE, evaluation label or holdout is opened. No fitting or target-agent calls.",
            "meaning": "Quote-anchored noisy task/field hypotheses, not permission, semantic truth or a risk score.",
            "failure": "Permanent pre-send claim; no retries. Failed, invalid and uncertain attempts remain explicitly unavailable with all-unclear fallback.",
            "capture": "Only public assistant message output and model/status/id/timing/usage metadata; never hidden reasoning.",
            "identity": "Repeated response IDs are excluded from annotation delivery; they are not separate successes."}}
    plan["sha256"] = digest(plan)
    durable_new(out / "PLAN.json", plan)
    return plan


def load_plan(out):
    out = Path(out)
    plan = read(out / "PLAN.json")
    if (plan.get("schema") != "submission-initial-semantics-v1"
            or digest({k: v for k, v in plan.items() if k != "sha256"}) != plan.get("sha256")):
        raise ValueError("semantic PLAN changed")
    if plan["api"] != API or set(plan["source_sha256"]) != set(SOURCES):
        raise ValueError("frozen API/source contract changed")
    if (plan["prompt"] != PROMPT or plan["schema_definition"] != SCHEMA
            or plan["normalization_shapes"] != [list(item) for item in SHAPES]):
        raise ValueError("semantic prompt/schema changed")
    for name, expected in plan["source_sha256"].items():
        if sha(out / "source" / name) != expected or sha(ROOT / name) != expected:
            raise ValueError("frozen semantic source changed")
    parent_path = Path(plan["parent_plan_path"])
    if sha(parent_path) != plan["parent_plan_file_sha256"]:
        raise ValueError("parent PLAN bytes changed")
    parent, contexts, bindings = initial_contexts(parent_path)
    if parent["sha256"] != plan["parent_plan_sha256"] or bindings != plan["bindings"]:
        raise ValueError("parent initial context binding changed")
    if set(contexts) != set(plan["contexts_sha256"]):
        raise ValueError("complete sixteen-context inventory required")
    for identity, expected in plan["contexts_sha256"].items():
        path = out / "contexts" / (identity + ".json")
        if sha(path) != expected or read(path) != contexts[identity]:
            raise ValueError("frozen initial context changed")
    return plan


def canonical_response(raw):
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json")
    if not isinstance(raw, dict):
        raise ValueError("response object required")
    result = {key: copy.deepcopy(raw.get(key)) for key in
              ("id", "model", "status", "created_at", "completed_at", "usage", "incomplete_details")}
    result["output"] = copy.deepcopy([item for item in raw.get("output", [])
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant"])
    result["capture_policy"] = "assistant_message_output_only; reasoning/tool-output items omitted"
    return result


def classification(out, identity, plan):
    """Rebuild from immutable claims/receipts, not from stored classifications."""
    out, name = Path(out), identity + ".json"
    context = read(out / "contexts" / name)
    row = {"schema": "submission-semantic-classification-v1", "plan_sha256": plan["sha256"],
        "context_sha256": identity, "status": "not_started", "attempted": False,
        "request_seq": None, "attempted_at": None, "received_at": None,
        "response_id": None, "response_status": None, "response_model": None,
        "response_created_at": None, "usage": None, "record_id": None,
        "annotation": unknown_annotation(context), "fallback_used": True,
        "semantic_truth_verified": False, "grants_permission": False}
    attempt_path = out / "attempts" / name
    if not attempt_path.exists():
        if (out / "responses" / name).exists() or (out / "failures" / name).exists():
            raise ValueError("response/failure without durable pre-send claim")
        return row
    claim = read(attempt_path)
    expected = request_payload(context)
    if (claim.get("plan_sha256") != plan["sha256"] or claim.get("context_sha256") != identity
            or claim.get("request") != expected or claim.get("request_sha256") != digest(expected)):
        raise ValueError("semantic claim differs from frozen context/request")
    attempted_at = datetime.fromisoformat(claim["attempted_at"])
    row.update(status="attempted_unknown", attempted=True, request_seq=claim["request_seq"],
               attempted_at=claim["attempted_at"])
    failure_path, response_path = out / "failures" / name, out / "responses" / name
    if failure_path.exists():
        row["failure"] = read(failure_path)
        if (row["failure"].get("request_sha256") != digest(expected)
                or datetime.fromisoformat(row["failure"]["at"]) < attempted_at):
            raise ValueError("failure binding or chronology changed")
        row["status"] = "api_error"
    if not response_path.exists():
        return row
    receipt = read(response_path)
    if (receipt.get("request_sha256") != digest(expected)
            or receipt.get("plan_sha256") != plan["sha256"]
            or receipt.get("context_sha256") != identity
            or datetime.fromisoformat(receipt["received_at"]) < attempted_at):
        raise ValueError("response binding or chronology changed")
    response = receipt["response"]
    row.update(received_at=receipt["received_at"], response_id=response.get("id"),
        response_status=response.get("status"), response_model=response.get("model"),
        response_created_at=response.get("created_at"), usage=response.get("usage"))
    if "failure" in row:
        row["status"] = "failed_after_response"
        return row
    try:
        if (not isinstance(response.get("id"), str) or not response["id"].strip()
                or response.get("model") != API["model"]):
            raise ValueError("nonempty response ID and exact requested model required")
        annotation = response_annotation(response)
    except (ValueError, TypeError, KeyError, AttributeError):
        row["status"] = "invalid_response"
        return row
    try:
        annotation = validate_annotation(context, annotation)
    except (ValueError, TypeError, KeyError, AttributeError):
        row["status"] = "invalid_annotation"
        return row
    row.update(status="quote_anchored_hypothesis", annotation=annotation, fallback_used=False,
        record_id="submission-semantics:" + digest({"plan": plan["sha256"], "context": identity,
            "response_id": response["id"], "receipt_sha256": sha(response_path)}))
    return row


def verify(out=DEFAULT_OUTPUT, *, check_derived=True):
    out = Path(out)
    plan = load_plan(out)
    identities = set(plan["contexts_sha256"])
    for folder in ("attempts", "responses", "failures", "classifications"):
        if any(path.stem not in identities for path in (out / folder).glob("*.json")):
            raise ValueError("unexpected context artifact outside fixed budget")
    rows = [classification(out, identity, plan) for identity in sorted(identities)]
    attempts = [row for row in rows if row["attempted"]]
    sequences = [row["request_seq"] for row in attempts]
    if (len(attempts) > 16 or any(type(v) is not int for v in sequences)
            or sorted(sequences) != list(range(1, len(attempts) + 1))):
        raise ValueError("unique contiguous pre-send sequence and sixteen-call budget required")
    response_ids = Counter(row["response_id"] for row in rows if isinstance(row["response_id"], str) and row["response_id"])
    excluded = sorted(row["context_sha256"] for row in rows
                      if isinstance(row["response_id"], str) and response_ids[row["response_id"]] > 1)
    for row in rows:
        target = out / "classifications" / (row["context_sha256"] + ".json")
        if check_derived and target.exists() and read(target) != row:
            raise ValueError("stored classification differs from immutable receipts")
    return {"schema": "submission-semantic-collection-v1", "plan_sha256": plan["sha256"],
        "parent_plan_sha256": plan["parent_plan_sha256"], "api_attempts": len(attempts),
        "maximum_api_calls": 16, "statuses": dict(Counter(row["status"] for row in rows)),
        "usable_hypotheses": sum(row["status"] == "quote_anchored_hypothesis" and row["context_sha256"] not in excluded for row in rows),
        "response_identity_exclusions": excluded,
        "contexts": [{"context_sha256": row["context_sha256"], "status": row["status"],
                      "fallback_used": row["fallback_used"], "identity_eligible": row["context_sha256"] not in excluded} for row in rows],
        "new_target_agent_calls": 0, "fits": 0, "semantic_truth_verified": False}


async def collect(out, transport):
    """Injected fake transport in tests. Only run_api instantiates an SDK client."""
    out = Path(out)
    plan = load_plan(out)
    with collection_lock(out):
        previous = verify(out, check_derived=False)
        sequence = [previous["api_attempts"]]
        semaphore = asyncio.Semaphore(API["concurrency"])

        async def one(identity):
            async with semaphore:
                row = classification(out, identity, plan)
                name = identity + ".json"
                if not row["attempted"]:
                    context = read(out / "contexts" / name)
                    request = request_payload(context)
                    sequence[0] += 1
                    if sequence[0] > API["maximum_api_calls"]:
                        raise ValueError("frozen call budget exhausted")
                    durable_new(out / "attempts" / name, {"plan_sha256": plan["sha256"],
                        "context_sha256": identity, "request_seq": sequence[0], "attempted_at": now(),
                        "request_sha256": digest(request), "request": request})
                    try:
                        raw = await asyncio.wait_for(transport(copy.deepcopy(request)), timeout=API["timeout_seconds"])
                        response = canonical_response(raw)
                        durable_new(out / "responses" / name, {"plan_sha256": plan["sha256"],
                            "context_sha256": identity, "request_sha256": digest(request),
                            "received_at": now(), "response": response})
                    except Exception as exc:
                        durable_new(out / "failures" / name, {"at": now(), "request_sha256": digest(request),
                            "error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None),
                            "retry_allowed": False})
                    row = classification(out, identity, plan)
                replace_summary(out / "classifications" / name, row)
                print(json.dumps({"context": identity, "status": row["status"],
                                  "attempts": int(row["attempted"])}), flush=True)

        await asyncio.gather(*(one(identity) for identity in sorted(plan["contexts_sha256"])))
        result = verify(out)
        replace_summary(out / "COLLECTION.json", result)
        return result


def annotation_for_case(out, case_id):
    """Exact project_prefix interface; unavailable or duplicate receipts yield None."""
    out = Path(out)
    plan, result = load_plan(out), verify(out)
    bindings = [row for row in plan["bindings"] if row["case_id"] == case_id]
    if len(bindings) != 1:
        raise ValueError("one frozen case binding required")
    identity = bindings[0]["context_sha256"]
    row = classification(out, identity, plan)
    if row["status"] != "quote_anchored_hypothesis" or identity in result["response_identity_exclusions"]:
        return None
    return {"context": read(out / "contexts" / (identity + ".json")),
            "annotation": row["annotation"], "record_id": row["record_id"],
            "availability": "offline_initial_only"}


def validate_endpoint(endpoint):
    value = urlsplit(endpoint)
    if (value.scheme != "http" or value.hostname not in ("localhost", "127.0.0.1")
            or value.port != 58661 or value.path.rstrip("/") != "/v1"
            or value.username or value.password or value.query or value.fragment):
        raise ValueError("authorized local endpoint http://localhost:58661/v1 required")
    return endpoint.rstrip("/")


async def run_api(out, endpoint):
    endpoint = validate_endpoint(endpoint)
    load_plan(out)
    key = os.environ.get("PAPERALPHA_LOCAL_API_KEY")
    if not key:
        raise ValueError("process-local PAPERALPHA_LOCAL_API_KEY required")
    from openai import AsyncOpenAI
    async with AsyncOpenAI(base_url=endpoint, api_key=key, max_retries=0, timeout=90) as client:
        async def transport(request):
            return await client.responses.create(**request)
        return await collect(out, transport)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent-plan", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--endpoint", default="http://localhost:58661/v1")
    args = parser.parse_args()
    if args.mode == "prepare":
        value = prepare(args.out, args.parent_plan)
        result = {"prepared": str(args.out.resolve()), "contexts": len(value["contexts_sha256"]),
                  "maximum_api_calls": 16, "plan_sha256": value["sha256"]}
    elif args.mode == "verify":
        result = verify(args.out)
    else:
        result = asyncio.run(run_api(args.out, args.endpoint))
    print(json.dumps(result, ensure_ascii=False))
