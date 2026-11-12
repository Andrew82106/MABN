"""One attempt at the eight PREVIOUSLY frozen test cases, scored before release.

No fitting, model changes, prompt changes, replacement cases or real publication.
All learned scoring is isolated in a numerical subprocess without credentials.
"""
import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_privacy_transfer_api import Journal, write, sha, now
from paperalpha_runtime.canonical import digest
from paperalpha_runtime.privacy_transfer import delivered_text
from paperalpha_runtime.branched_publisher_data import issuance, transitions, stage_request, prefix, outcome, literal_profile

BN_PYTHON = r"D:\anaconda\envs\multi_agent_graph\python.exe"
SOURCES = ('code/commands/collect_branch_publisher_test.py', 'code/commands/monitor_branch_publisher.py',
    'code/commands/verify_branch_publisher_test.py', 'code/tests/test_branch_publisher_test_protocol.py')


def prepare(data_run, fit, audit, out):
    if out.exists():
        raise ValueError("new held-out run directory required")
    data = json.loads((data_run / "PLAN.json").read_text(encoding="utf-8"))
    if data["sha256"] != digest({k: v for k, v in data.items() if k != "sha256"}):
        raise ValueError("original task plan changed")
    fit_plan = json.loads((fit / "PLAN.json").read_text(encoding="utf-8"))
    frozen = json.loads((fit / "FROZEN.json").read_text(encoding="utf-8"))
    verified = json.loads((audit / "REPORT.json").read_text(encoding="utf-8"))
    if fit_plan["data_plan_sha256"] != sha(data_run / "PLAN.json") or verified["report_sha256"] != sha(fit / "REPORT.json") or not verified["selection_recomputed"]:
        raise ValueError("fit and audit must bind original task plan")
    report = json.loads((fit / "REPORT.json").read_text(encoding="utf-8"))
    for name, expected in report["artifacts_sha256"].items():
        if sha(fit / name) != expected:
            raise ValueError("fitted experiment changed: " + name)
    for name, expected in data["sources_sha256"].items():
        if sha(data_run / "source_snapshot" / name) != expected:
            raise ValueError("frozen acquisition changed")
    selected_cases = [c for c in data["cases"] if c["split"] == "test"]
    if len(selected_cases) != 8 or len({c["template_id"] for c in selected_cases}) != 2:
        raise ValueError("exact original eight-case test group required")
    for name in ("FROZEN.json", *frozen["files_sha256"]):
        target = out / "bundle" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fit / name, target)
    files = {p.relative_to(fit / "source_snapshot").as_posix(): p for p in (fit / "source_snapshot").rglob("*") if p.is_file()}
    for name in data["sources_sha256"]:
        if name.startswith('code/src/paperalpha_runtime/'):
            if name in files and sha(files[name]) != sha(data_run / "source_snapshot" / name):
                raise ValueError("model and acquisition runtime snapshots differ")
    files.update({s: ROOT / s for s in SOURCES})
    for name, source in files.items():
        target = out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name, source in (("original_task_plan.json", data_run / "PLAN.json"), ("fit_audit.json", audit / "REPORT.json")):
        shutil.copyfile(source, out / name)
    plan = {"schema": "branch-publisher-heldout-v1", "created_at": now(), "cases": selected_cases,
        "settings": data["settings"], "bn_python": BN_PYTHON,
        "source_data_run": str(data_run), "source_fit": str(fit), "source_fit_audit": str(audit),
        "original_plan_sha256": sha(out / "original_task_plan.json"), "fit_report_sha256": sha(fit / "REPORT.json"),
        "fit_audit_sha256": sha(out / "fit_audit.json"), "bundle_sha256": sha(out / "bundle/FROZEN.json"),
        "source_sha256": {name: sha(out / "source_snapshot" / name) for name in files},
        "maximum_api_calls": 32,
        "protocol": {"cutoff": "All methods scored on committed prefix before publisher API dispatch; scores never sent to agents",
            "same_tasks": "Exactly the eight test configurations frozen before development acquisition; unchanged prompts/settings/runtime",
            "no_tuning": "No fit, threshold changes, retries, replacements or case edits. One attempt per case; failures remain failures",
            "metrics": "Frozen k=1 primary / k=2 secondary, Brier/NLL clipped1e-12/fixed0.5/F1/AUROC/AP; completed runs only, retain failed denominator",
            "selection": "Report every frozen selected family, no test-based candidate reselection",
            "scope": "Eight related-template held-out runs, not broad independent MAS benchmark or original method reproduction"},
        "docs": ["https://developers.openai.com/api/docs/models/gpt-5.5", "https://developers.openai.com/api/docs/guides/reasoning#allocating-space-for-reasoning"]}
    plan["sha256"] = digest(plan)
    write(out / "PLAN.json", plan)
    print(json.dumps({"prepared": str(out), "heldout_cases": len(selected_cases), "max_api_calls": 32}), flush=True)


def load(out):
    plan = json.loads((out / "PLAN.json").read_text(encoding="utf-8"))
    if plan["schema"] != "branch-publisher-heldout-v1" or plan["sha256"] != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("heldout plan changed")
    for name, expected in plan["source_sha256"].items():
        if sha(out / "source_snapshot" / name) != expected:
            raise ValueError("heldout code changed")
    for name, expected in (("original_task_plan.json", plan["original_plan_sha256"]), ("fit_audit.json", plan["fit_audit_sha256"]), ("bundle/FROZEN.json", plan["bundle_sha256"])):
        if sha(out / name) != expected:
            raise ValueError("frozen provenance changed")
    original = json.loads((out / "original_task_plan.json").read_text(encoding="utf-8"))
    if plan["cases"] != [c for c in original["cases"] if c["split"] == "test"] or plan["settings"] != original["settings"]:
        raise ValueError("heldout configurations or settings differ")
    frozen = json.loads((out / "bundle/FROZEN.json").read_text(encoding="utf-8"))
    for name, expected in frozen["files_sha256"].items():
        if sha(out / "bundle" / name) != expected:
            raise ValueError("selected model changed")
    return plan


class Monitor:
    def __init__(self, out, python):
        env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "PAPERALPHA_API_KEY", "CODEX_API_KEY"):
            env.pop(key, None)
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.child = subprocess.Popen([python, "-X", "utf8", "-B", "-u", str(out / "source_snapshot/code/commands/monitor_branch_publisher.py"),
            "--bundle", str(out / "bundle")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        line = self.child.stdout.readline()
        if not line or not json.loads(line).get("ready"):
            self.child.wait(timeout=20)
            raise RuntimeError("isolated frozen monitor startup failed")

    def score(self, message):
        self.child.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.child.stdin.flush()
        line = self.child.stdout.readline()
        if not line or not (reply := json.loads(line)).get("ok"):
            raise RuntimeError("frozen monitor rejected prefix")
        return reply["result"]

    def close(self):
        self.child.stdin.close()
        code = self.child.wait(timeout=20)
        if code or self.child.stderr.read().strip():
            raise RuntimeError("frozen monitor closed with errors")


async def collect(out, endpoint):
    from openai import AsyncOpenAI
    plan = load(out)
    if Path(__file__).resolve() != out / "source_snapshot/code/commands/collect_branch_publisher_test.py":
        raise ValueError("run frozen heldout collector")
    url = urlparse(endpoint)
    if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1"} or url.path.rstrip("/") != "/v1" or url.username or url.password or url.query or url.fragment:
        raise ValueError("explicit authorized local API endpoint required")
    key = os.environ.get("PAPERALPHA_API_KEY")
    if not key:
        raise ValueError("process-local credential required")
    destination = out / "test"
    destination.mkdir(exist_ok=False)
    journal = Journal(destination / "journal.jsonl")
    monitor = Monitor(out, plan["bn_python"])
    settings = plan["settings"]
    journal.add("start", plan_sha256=plan["sha256"], endpoint=endpoint, python=sys.version,
        openai_version=importlib.metadata.version("openai"), bundle_sha256=plan["bundle_sha256"])
    case_slots, api_slots, monitor_lock = asyncio.Semaphore(settings["case_concurrency"]), asyncio.Semaphore(settings["api_concurrency"]), asyncio.Lock()
    try:
        async with AsyncOpenAI(base_url=endpoint, api_key=key, max_retries=0, timeout=settings["timeout_seconds"]) as client:
            async def one(case):
                async with case_slots:
                    case_id = case["id"]
                    record = {"id": case_id, "split": "test", "status": "started", "calls": [], "failures": []}
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
                                return {"ok": True, **delivered_text(raw)}
                            except Exception as exc:
                                failure = journal.add("call_failure", case_id, stage=stage, request_seq=sent["seq"],
                                    error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
                                record["failures"].append(failure["seq"])
                                return {"ok": False}

                    planner = await call("planner", stage_request(case, "planner"))
                    replies = {"planner": planner}
                    if planner["ok"]:
                        workers_result = await asyncio.gather(*(call(name, stage_request(case, name, planner_text=planner["text"])) for name in ("worker-a", "worker-b")))
                        replies.update(zip(("worker-a", "worker-b"), workers_result))
                        if all(reply["ok"] for reply in workers_result):
                            workers = {name: replies[name]["text"] for name in ("worker-a", "worker-b")}
                            updates = transitions(case)
                            journal.add("authority_transition", case_id, records=updates)
                            actual = initial + updates
                            prepared = stage_request(case, "publisher", workers=workers)
                            ready = journal.add("prefix_ready", case_id, observation=prefix(case, planner["text"], workers, actual, prepared))
                            record["prefix_seq"] = ready["seq"]
                            prediction = None
                            try:
                                async with monitor_lock:
                                    prediction = await asyncio.to_thread(monitor.score, {"observation": ready["observation"], "prefix_sequence": ready["seq"]})
                            except Exception as exc:
                                failure = journal.add("monitor_failure", case_id, error_type=type(exc).__name__, prefix_seq=ready["seq"])
                                record["failures"].append(failure["seq"])
                            if prediction is not None:
                                saved = journal.add("prediction", case_id, prefix_seq=ready["seq"], result=prediction)
                                record["prediction_seq"] = saved["seq"]
                                publisher = await call("publisher", prepared)
                                replies["publisher"] = publisher
                                if publisher["ok"]:
                                    final = journal.add("outcome", case_id, outcome=outcome(case, actual, publisher["text"]),
                                        literal_profile=literal_profile(case, planner["text"], workers, publisher["text"]))
                                    record.update(status="completed", outcome_seq=final["seq"])
                    if record["status"] != "completed":
                        record["status"] = "failed"
                    record["native_refusal_stages"] = [name for name, reply in replies.items() if reply["ok"] and reply["native_refusal"]]
                    record["calls"].sort(key=lambda call: call["request_seq"])
                    record["failures"].sort()
                    journal.add("case_complete", case_id, status=record["status"])
                    write(destination / "cases" / (case_id + ".json"), record)
                    print(json.dumps({"case": case_id, "status": record["status"], "responses": len(record["calls"])}), flush=True)
                    return record
            records = await asyncio.gather(*(one(case) for case in plan["cases"]))
        monitor.close()
        journal.add("complete", cases=len(records), completed=sum(r["status"] == "completed" for r in records))
    finally:
        journal.close()
        if monitor.child.poll() is None:
            monitor.close()
    write(destination / "COLLECTION.json", {"created_at": now(), "plan_sha256": plan["sha256"], "cases": records,
        "artifact_sha256": {p.relative_to(destination).as_posix(): sha(p) for p in destination.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--data-run", type=Path)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://localhost:58661/v1")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.data_run.resolve(), args.fit.resolve(), args.audit.resolve(), args.out.resolve())
    else:
        asyncio.run(collect(args.out.resolve(), args.endpoint))
