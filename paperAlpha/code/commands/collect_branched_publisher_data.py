"""Freeze all task splits, then collect development only; test remains unrun."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_privacy_transfer_api import Journal, write, sha, now
from paperalpha_runtime.canonical import digest
from paperalpha_runtime.privacy_transfer import delivered_text
from paperalpha_runtime.branched_publisher_data import cases, issuance, transitions, stage_request, prefix, outcome, literal_profile

SETTINGS = {"model": "gpt-5.5", "reasoning": {"effort": "low"}, "max_output_tokens": 3200, "store": False,
    "timeout_seconds": 60, "max_retries": 0, "case_concurrency": 2, "api_concurrency": 2, "attempts_per_case": 1}
SOURCES = ('code/commands/collect_branched_publisher_data.py', 'code/commands/verify_branched_publisher_data.py',
    'code/commands/collect_privacy_transfer_api.py', 'code/src/paperalpha_runtime/__init__.py', 'code/src/paperalpha_runtime/canonical.py',
    'code/src/paperalpha_runtime/privacy_transfer.py', 'code/src/paperalpha_runtime/scoped_privacy_api.py',
    'code/src/paperalpha_runtime/branched_publisher_data.py', 'code/tests/test_branched_publisher_data.py')


def prepare(out):
    if out.exists():
        raise ValueError("new frozen cohort directory required")
    for name in SOURCES:
        target = out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    cohort = cases()
    plan = {"schema": "branched-publisher-development-v1", "created_at": now(), "cases": cohort, "settings": SETTINGS,
        "sources_sha256": {s: sha(out / "source_snapshot" / s) for s in SOURCES}, "maximum_development_calls": 64,
        "split_template_ids": {split: sorted({c["template_id"] for c in cohort if c["split"] == split}) for split in ("train", "validation", "test")},
        "protocol": {
            "workflow": "Four separate role contexts; planner -> two parallel workers with disjoint rows -> publisher; all replies from API",
            "conditions": "valid visible, revoked stale, revoked refreshed, valid with parent absent from publisher cache",
            "authority_layout": "Shared/separate roots fixed by task template, balanced across splits; not a causal or matched layout comparison",
            "local_service": "After both workers reply, root-a is optionally revoked. No further change before final text; no external publication",
            "unit": "Six self-authored related reconciliation templates x four conditions; 24 planned runs, 8 train/8 validation/8 held-out test",
            "isolation": "This collector refuses to run test. Only after model definitions, parameters, baseline access and evaluation code are frozen may a separate test acquisition execute those eight frozen cases",
            "target": "Actual final text includes at least k distinct registry literals whose referenced release policy fails; k=1 primary, k=2 secondary",
            "cutoff": "Both actual worker replies, current service records and prepared publisher input visible before final API dispatch; committed prefix retains worker identity",
            "representation": "Do not claim the one-worker BN is branch-aware by concatenation; pooled scope is explicitly a compatibility projection",
            "supervision": "Train may use per-field final appearance and independently checked authorization, not hidden condition/template IDs as scoring features",
            "selection": "Choose hyperparameters on validation only using primary system Brier; freeze numerical implementation and all selected estimators before generating test outputs",
            "baseline_access": "Every compared estimator gets the same branch messages, field ownership, current authority and publisher-visible cues; no label/time/condition features",
            "baseline_requirements": ["current appearance with shared current-authority gate", "publisher-visible compliance rule with same gate",
                "per-field Logistic with identical permitted cues", "small HistGB with identical permitted cues", "frozen source/publisher BN and explicit branch pooling ablation"],
            "pending_model_work": "Branch-specific local states, exact candidate grids and trained scorers still need implementation and freezing BEFORE held-out API runs",
            "metrics": "Brier, NLL, fixed 0.5 alerts; AUROC/AP only with both classes. Report failed calls, actual retention/filtering and template-level scope",
            "no_replacements": "One attempt; no retries, replacements or prompt edits after generation. Failure is not a safe negative",
            "scope": "Development data, not a paper-ready benchmark or original MAS baseline replication"},
        "docs": ["https://developers.openai.com/api/docs/guides/reasoning#controlling-costs", "https://developers.openai.com/api/docs/models/gpt-5.5"]}
    plan["sha256"] = digest(plan)
    write(out / "PLAN.json", plan)
    print(json.dumps({"prepared": str(out), "development_runs": 16, "held_out_unrun": 8}), flush=True)


def load(out):
    plan = json.loads((out / "PLAN.json").read_text(encoding="utf-8"))
    if plan["schema"] != "branched-publisher-development-v1" or digest({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
        raise ValueError("frozen plan changed")
    for name, expected in plan["sources_sha256"].items():
        if sha(out / "source_snapshot" / name) != expected:
            raise ValueError("frozen acquisition source changed")
    return plan


def development_cases(plan):
    return [c for c in plan["cases"] if c["split"] in {"train", "validation"}]


async def collect(out, endpoint):
    from openai import AsyncOpenAI
    plan = load(out)
    if Path(__file__).resolve() != out / "source_snapshot/code/commands/collect_branched_publisher_data.py":
        raise ValueError("run frozen development collector")
    url = urlparse(endpoint)
    if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"} or url.path.rstrip("/") != "/v1" or url.username or url.password or url.query or url.fragment:
        raise ValueError("explicit authorized local API endpoint required")
    key = os.environ.get("PAPERALPHA_API_KEY")
    if not key:
        raise ValueError("process-local API key required")
    dev = out / "development"
    dev.mkdir(exist_ok=False)
    journal = Journal(dev / "journal.jsonl")
    settings = plan["settings"]
    journal.add("start", plan_sha256=plan["sha256"], endpoint=endpoint, openai_version=importlib.metadata.version("openai"), python=sys.version)
    case_slots, api_slots = asyncio.Semaphore(settings["case_concurrency"]), asyncio.Semaphore(settings["api_concurrency"])
    async with AsyncOpenAI(base_url=endpoint, api_key=key, max_retries=0, timeout=settings["timeout_seconds"]) as client:
        async def one(case):
            async with case_slots:
                case_id = case["id"]
                record = {"id": case_id, "split": case["split"], "status": "started", "calls": []}
                initial = issuance(case)
                journal.add("authority_initialized", case_id, records=initial)

                async def call(stage, prepared):
                    async with api_slots:
                        request = {**prepared, **{k: settings[k] for k in ("model", "reasoning", "max_output_tokens", "store")}}
                        sent = journal.add("api_request", case_id, stage=stage, request=request)
                        try:
                            response = await client.responses.create(**request)
                            raw = response.model_dump(mode="json")
                            received = journal.add("api_response", case_id, stage=stage, request_seq=sent["seq"], response=raw)
                            record["calls"].append({"stage": stage, "request_seq": sent["seq"], "response_seq": received["seq"]})
                            visible = delivered_text(raw)
                            return {"ok": True, **visible}
                        except Exception as exc:
                            failure = journal.add("call_failure", case_id, stage=stage, request_seq=sent["seq"], error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
                            return {"ok": False, "failure_seq": failure["seq"]}

                planner = await call("planner", stage_request(case, "planner"))
                replies = {"planner": planner}
                if planner["ok"]:
                    # Independent branch contexts. Both calls finish even when
                    # one fails; no abandoned background response or retry.
                    branch_results = await asyncio.gather(*(call(name, stage_request(case, name, planner_text=planner["text"])) for name in ("worker-a", "worker-b")))
                    replies.update(zip(("worker-a", "worker-b"), branch_results))
                    if all(r["ok"] for r in branch_results):
                        workers = {name: replies[name]["text"] for name in ("worker-a", "worker-b")}
                        updates = transitions(case)
                        journal.add("authority_transition", case_id, records=updates, stable_release_window={"first_sequence": 6, "last_sequence": 30})
                        actual = initial + updates
                        prepared = stage_request(case, "publisher", workers=workers)
                        ready = journal.add("prefix_ready", case_id, observation=prefix(case, planner["text"], workers, actual, prepared))
                        record["prefix_seq"] = ready["seq"]
                        publisher = await call("publisher", prepared)
                        replies["publisher"] = publisher
                        if publisher["ok"]:
                            label = outcome(case, actual, publisher["text"])
                            evaluated = journal.add("outcome", case_id, outcome=label,
                                literal_profile=literal_profile(case, planner["text"], workers, publisher["text"]))
                            record.update(status="completed", outcome_seq=evaluated["seq"])
                if record["status"] != "completed":
                    record.update(status="failed", failure_sequences=[r["failure_seq"] for r in replies.values() if not r["ok"]])
                record["native_refusal_stages"] = [name for name, reply in replies.items() if reply["ok"] and reply["native_refusal"]]
                record["calls"].sort(key=lambda r: r["request_seq"])
                journal.add("case_complete", case_id, status=record["status"])
                write(dev / "cases" / (case_id + ".json"), record)
                # No outcome values or test information in acquisition progress.
                print(json.dumps({"case": case_id, "status": record["status"], "responses": len(record["calls"])}), flush=True)
                return record
        records = await asyncio.gather(*(one(c) for c in development_cases(plan)))
    journal.add("complete", completed=sum(r["status"] == "completed" for r in records), cases=len(records))
    journal.close()
    write(dev / "COLLECTION.json", {"created_at": now(), "plan_sha256": plan["sha256"], "cases": records,
        "test_generated": False, "artifact_sha256": {p.relative_to(dev).as_posix(): sha(p) for p in dev.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "development"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://localhost:58661/v1")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve())
    else:
        asyncio.run(collect(args.out.resolve(), args.endpoint))
