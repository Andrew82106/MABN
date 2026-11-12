"""Prepare/verify four fixed shared-credit development cases; run is opt-in.

prepare and verify are fully offline. No old collector, PLAN or results are
changed. The complete source/fixture manifest and fixed prompts are frozen
before any API call. A durable run claim prevents retries or replenishment.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from paperalpha_runtime.shared_credit_pilot import (
    SCHEMA, SETTINGS, SYSTEM, SharedCreditPilot, canonical, digest, load_bank,
    planned_cases, public_ledger, snapshot, tool_definitions, write_new,
)

UPSTREAM_COMMIT = "6317162590aeeb1c8dde32b880ac199933343e4a"
DEFAULT_UPSTREAM = ROOT / 'code/external/HarnessAudit'
DEFAULT_OUTPUT = ROOT / 'experiments/harnessaudit/runs/submission__shared_credit_pilot'
LOCAL_SOURCES = ('code/src/paperalpha_runtime/shared_credit_pilot.py', 'code/src/paperalpha_runtime/__init__.py',
                 'code/commands/collect_shared_credit_pilot.py', 'code/tests/test_shared_credit_pilot.py',
                 'code/tests/test_shared_credit_collection.py')


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _upstream_sources(upstream):
    paths = ["LICENSE", "multi_agent/banks/ecommerce.py", "multi_agent/tools/ecommerce.yaml"]
    paths += [p.relative_to(upstream).as_posix() for p in sorted((upstream/"multi_agent/fixtures/ecommerce").rglob("*.json"))]
    if len(paths) <= 3:
        raise ValueError("author seed inventory required")
    return paths


def _check_upstream(upstream, names):
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.run(["git", "-C", str(upstream), "diff", "--exit-code", UPSTREAM_COMMIT, "--", *names], capture_output=True)
    if head != UPSTREAM_COMMIT or changed.returncode:
        raise ValueError("unchanged pinned author Bank/tools/seeds required")


def prepare(output=DEFAULT_OUTPUT, upstream=DEFAULT_UPSTREAM):
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    if output.exists():
        raise ValueError("pilot directory already exists; never overwrite a plan or attempt")
    names = _upstream_sources(upstream)
    _check_upstream(upstream, names)
    cases = planned_cases()
    bank = load_bank(upstream)
    try:
        # This only initializes a private :memory: seed; no model or financial call.
        initial = public_ledger(bank, cases[0]["policy"], 0)
        if initial["recorded_store_credit_cents"] != 0:
            raise ValueError("reviewed pilot expects this seeded scoped ledger to be empty")
        order = bank.conn.execute("SELECT customer_id FROM orders WHERE order_id=?", (cases[0]["policy"]["binding"]["order_id"],)).fetchone()
        if order is None or order[0] != cases[0]["policy"]["binding"]["customer_id"]:
            raise ValueError("seeded scoped order/customer binding required")
    finally:
        bank.close()
    tools = tool_definitions(upstream)
    files = {"local/"+p: ROOT/p for p in LOCAL_SOURCES}
    files.update({"upstream/"+p: upstream/p for p in names})
    for name, path in files.items():
        if not path.is_file(): raise ValueError("missing frozen source: "+name)
    output.mkdir(parents=True)
    manifest = {}
    for name, path in files.items():
        destination = output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[name] = sha(destination)
    plan = {"schema": SCHEMA, "purpose": "four-case mechanism development only; no model superiority or publication claim",
            "upstream": str(upstream), "upstream_commit": UPSTREAM_COMMIT, "source_sha256": manifest,
            "settings": SETTINGS, "cases": cases, "system_instruction": SYSTEM, "tools": tools,
            "environment": {"python": sys.version, "utf8_mode": sys.flags.utf8_mode,
                            "libraries": {name: importlib.metadata.version(name) for name in ("openai", "PyYAML")}},
            "tool_budget": "Only schema-valid calls offered execution consume the 6-per-instance budget; all proposals, including rejected schemas and over-budget calls, count as attempted_tool_invocations.",
            "order_identity": "Pinned author writer strips whitespace and prefixes # on every ledger insert; original fixtures seed no ledger rows. Public and private scopes use the same customer and normalized order.",
            "schedule": "Four cases executed once in fixed order; independent API contexts; same-round requests concurrent, all Bank calls serialized A then B in response order.",
            "visibility": "Both agents and monitor receive the same public scoped ledger. Only public/ is a monitor input. Private request reasoning, snapshot deltas, labels and manifests are not.",
            "failure": "All four cases remain in the denominator. Each API attempt is recorded before dispatch; no retries, resumes or outcome-based replenishment.",
            "policy": "Our explicit cumulative commitment extension; not original goodwill semantics, native HarnessAudit benchmark or actual settlement.",
            "future_baseline": "Same-information global exact accumulator is mandatory; per-agent checks are only an ablation.",
            "endpoint": "http://localhost:58661/v1", "api_calls_at_prepare": 0}
    plan["sha256"] = digest(plan)
    write_new(output/"PLAN.json", plan)
    verify(output)
    return {"plan_sha256": plan["sha256"], "cases": 4, "maximum_requests": 48, "api_calls": 0}


def verify(output):
    output = Path(output).resolve()
    plan = read(output/"PLAN.json")
    if (plan.get("schema") != SCHEMA or plan.get("sha256") != digest({k:v for k,v in plan.items() if k != "sha256"})
            or plan.get("settings") != SETTINGS or plan.get("cases") != planned_cases() or plan.get("system_instruction") != SYSTEM):
        raise ValueError("frozen plan/settings changed")
    upstream = Path(plan["upstream"])
    names = _upstream_sources(upstream)
    _check_upstream(upstream, names)
    expected = {"local/"+p: ROOT/p for p in LOCAL_SOURCES} | {"upstream/"+p: upstream/p for p in names}
    if set(plan["source_sha256"]) != set(expected):
        raise ValueError("source/seed inventory differs")
    for name, path in expected.items():
        if sha(path) != plan["source_sha256"][name] or sha(output/"source"/name) != plan["source_sha256"][name]:
            raise ValueError("frozen source changed: "+name)
    if plan["tools"] != tool_definitions(upstream):
        raise ValueError("frozen actual tool definitions changed")
    if plan["environment"] != {"python": sys.version, "utf8_mode": sys.flags.utf8_mode,
                              "libraries": {name: importlib.metadata.version(name) for name in ("openai", "PyYAML")}}:
        raise ValueError("frozen SDK/library versions changed")
    return plan


async def collect_all(output, provider):
    """Fake providers are injected only by tests; production uses run_api."""
    output = Path(output).resolve()
    plan = verify(output)
    write_new(output/"private/RUN.claim.json", {"plan_sha256": plan["sha256"], "pid": os.getpid(),
                                                "rule": "exclusive one-shot; never clear automatically"})
    summaries, response_ids, interrupted = [], set(), None
    for case in plan["cases"]:
        if interrupted is not None:
            summaries.append({"case_id": case["case_id"], "completed": False, "status": "not_attempted",
                              "actual_requests": 0, "actual_tools": 0, "attempted_tool_invocations": 0,
                              "bank_dispatches": 0, "eligible_for_effectiveness": False,
                              "reason": "cohort stopped after prior exception; no safe label inferred"})
            continue
        bank, pilot = None, None
        try:
            folder = output/"cases"/case["case_id"]
            folder.mkdir(parents=True, exist_ok=False)
            bank = load_bank(plan["upstream"])
            pilot = SharedCreditPilot(bank, case, plan["tools"], folder, cohort_response_ids=response_ids)
            summary = await pilot.run(provider)
            summaries.append(summary)
        except BaseException as exc:
            interrupted = exc
            agents = list(pilot.agents.values()) if pilot is not None else []
            summaries.append({"case_id": case["case_id"], "completed": False, "status": "interrupted",
                              "error_type": type(exc).__name__, "eligible_for_effectiveness": False,
                              "actual_requests": sum(a.requests for a in agents),
                              "actual_tools": sum(a.tools for a in agents),
                              "attempted_tool_invocations": sum(a.attempted_tools for a in agents),
                              "bank_dispatches": sum(t["bank_dispatched"] for t in pilot.transitions) if pilot is not None else 0})
        finally:
            if bank is not None: bank.close()
    source_error = None
    try:
        verify(output)
    except Exception as exc:
        source_error = type(exc).__name__
        if interrupted is None: interrupted = exc
    result = {"schema": SCHEMA, "plan_sha256": plan["sha256"], "cases": summaries,
              "actual_requests": sum(s["actual_requests"] for s in summaries),
              "actual_tools": sum(s["actual_tools"] for s in summaries),
              "attempted_tool_invocations": sum(s["attempted_tool_invocations"] for s in summaries),
              "bank_dispatches": sum(s["bank_dispatches"] for s in summaries),
              "unique_actual_response_ids": len(response_ids), "source_verification_error": source_error,
              "eligible_for_effectiveness": source_error is None and interrupted is None and all(s["eligible_for_effectiveness"] for s in summaries),
              "completed_cases": sum(s["completed"] for s in summaries),
              "fixed_case_count": 4, "publication_ready": False, "performance_claimed": False}
    if result["actual_requests"] > 48: raise ValueError("cohort request budget exceeded")
    write_new(output/"private/COLLECTION.json", result)
    if interrupted is not None: raise interrupted
    return result


async def run_api(output, endpoint, expected_plan_sha256):
    parsed = urlparse(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"} or parsed.port != 58661
            or parsed.path.rstrip("/") != "/v1" or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("approved explicit loopback endpoint required")
    plan = verify(output)
    if expected_plan_sha256 != plan["sha256"]:
        raise ValueError("explicit reviewed plan SHA required")
    key = os.environ.get("PAPERALPHA_LOCAL_API_KEY")
    if not key: raise ValueError("environment-only API credential required")
    from openai import AsyncOpenAI
    async with AsyncOpenAI(api_key=key, base_url=endpoint, max_retries=0, timeout=SETTINGS["timeout_seconds"]) as client:
        async def provider(_actor, request):
            return await client.responses.create(**request)
        return await collect_all(output, provider)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--endpoint", default="http://localhost:58661/v1")
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()
    if args.mode == "prepare": result = prepare(args.output, args.upstream)
    elif args.mode == "verify": result = {"verified_plan_sha256": verify(args.output)["sha256"], "api_calls": 0}
    else: result = asyncio.run(run_api(args.output, args.endpoint, args.expected_plan_sha256))
    print(canonical(result))
