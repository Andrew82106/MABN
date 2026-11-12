"""Prospective API transfer: frozen forecasts before the actual buyer calls."""
from pathlib import Path
import argparse
import asyncio
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_exclusive_purchase import read, write, sha, now
from paperalpha_runtime import crossed_purchase as study
from paperalpha_runtime.privacy_transfer import delivered_text

NUMERIC_PYTHON = "D:/anaconda/envs/multi_agent_graph/python.exe"
MODEL_NAMES = [f"{kind}_{penalty}.json" for penalty in (.001, .03)
               for kind in ("attribute_bn", "standalone_full", "standalone_cost_only", "standalone_warranty_only")]


def prepare(out, model_source):
    source_plan, source_report = read(model_source / "PLAN.json"), read(model_source / "REPORT.json")
    if source_report["plan_sha256"] != sha(model_source / "PLAN.json"):
        raise ValueError("source model report/plan binding differs")
    source_fit = read(model_source / "FIT.json")
    if source_report["fit_sha256"] != sha(model_source / "FIT.json"):
        raise ValueError("source fitted models changed")
    # Preserve every model, feature and policy source file of the prior final
    # experiment. This study adds only a new cohort/collector/scorer/evaluator.
    for n, h in source_plan["source_sha256"].items():
        if sha(ROOT / n) != h: raise ValueError("previous frozen implementation changed: " + n)
    out.mkdir(parents=True, exist_ok=False)
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'code/src').rglob("*.py")]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'knowledge/templates').glob("*.json")]
    names += ['knowledge/risks/risk_catalog.json', 'code/commands/collect_exclusive_purchase.py', 'code/commands/evaluate_exclusive_purchase.py',
              'code/commands/evaluate_choice_attributes.py', 'code/commands/scalar_binary_graph.py', 'code/commands/score_crossed_purchase.py',
              'code/commands/collect_crossed_purchase.py', 'code/commands/verify_crossed_purchase.py', 'code/tests/test_crossed_purchase.py']
    for n in names:
        p = out / "source" / n; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / n, p)
    for n in MODEL_NAMES:
        if sha(model_source / "models" / n) != source_fit["model_sha256"][n]: raise ValueError("source model changed")
        p = out / "models" / n; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(model_source / "models" / n, p)
    plan = {"schema": "crossed-purchase-prospective-v1", "created_at_utc": now(), "cases": study.cases(),
            "source_sha256": {n: sha(out / "source" / n) for n in sorted(names)},
            "model_source": str(model_source), "model_source_plan_sha256": sha(model_source / "PLAN.json"),
            "model_source_report_sha256": sha(model_source / "REPORT.json"),
            "model_sha256": {n: sha(out / "models" / n) for n in MODEL_NAMES},
            "numeric_python": NUMERIC_PYTHON,
            "api": {"endpoint": "http://localhost:58661/v1", "model": "gpt-5.5", "effort": "low",
                    "max_output_tokens": 3200, "timeout_seconds": 60, "transport_retries": 0, "concurrent_flows": 2,
                    "max_generation_calls": len(study.cases()) * 3},
            "rule_comparators": ["lowest_cost", "longest_warranty", "priority_value"],
            "excluded_historical_methods": "Old cue BN depends on basic/enhanced suffix extraction, which does not support these neutral catalogues. Not counted as a failed strong baseline.",
            "protocol": ["No fitting, calibration, threshold selection or retry/resampling on this cohort.",
                "24 authored unseen flows: 2 domains x 3 briefs x 4 attribute profiles, neutral labels and balanced display order.",
                "After actual planner response, record prepared prefix and all 11 forecasts before issuing either buyer request.",
                "Actual API replies are parsed before distinct local simulated dispatch events. No real purchases or monitoring-based blocking.",
                "Future risk forecast conditions on unchanged prepared delivery and a valid completed command protocol; all failures remain reported, never negative labels.",
                "No planned result selection. Report every profile, priority, failure, abstention and same-information comparator.",
                "New cohort is a small controlled transfer study, not a public benchmark or native MAS superiority claim."],
            "openai_docs": "https://developers.openai.com/api/docs/guides/structured-outputs#json-mode",
            "new_fits": 0, "old_test_read": False, "new_default_selected": False}
    write(out / "PLAN.json", plan)
    print(json.dumps({"prepared": str(out), "flows": len(plan["cases"]), "plan_sha256": sha(out / "PLAN.json")}), flush=True)


def load(out):
    plan = read(out / "PLAN.json")
    if ROOT != (out / "source").resolve(): raise ValueError("run the frozen source")
    for n, h in plan["source_sha256"].items():
        if sha(ROOT / n) != h: raise ValueError("frozen source changed: " + n)
    for n, h in plan["model_sha256"].items():
        if sha(out / "models" / n) != h: raise ValueError("frozen model changed")
    if plan["cases"] != study.cases(): raise ValueError("planned cases changed")
    return plan


async def numeric_forecast(out, directory, plan):
    command = [plan["numeric_python"], "-X", "utf8", "-B", str(ROOT / 'code/commands/score_crossed_purchase.py'),
               "--prefix", str(directory / "PREFIX.json"), "--models", str(out / "models"), "--out", str(directory / "FORECAST.json")]
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    creationflags=0x08000000 if sys.platform == "win32" else 0)
    stdout, stderr = await process.communicate()
    write(directory / "NUMERIC_WORKER.json", {"command": command, "returncode": process.returncode,
          "stdout": stdout.decode("utf-8", errors="replace"), "stderr": stderr.decode("utf-8", errors="replace")})
    if process.returncode: raise RuntimeError("frozen numerical forecast failed before buyer calls")
    prediction = read(directory / "FORECAST.json")
    if prediction["model_sha256"] != plan["model_sha256"]: raise ValueError("forecast model identity differs")
    return prediction


async def run_flow(case, out, plan, ask_api, forecast=numeric_forecast):
    directory = out / "runs" / case["id"]; directory.mkdir(parents=True, exist_ok=False)
    records = []; calls = 0; complete = False; failure = None; prediction_made = False
    def record(value):
        r = {"sequence": len(records) + 1, "at_utc": now(), "monotonic_ns": time.monotonic_ns(), **value}
        write(directory / f"record_{len(records) + 1:04d}.json", r); records.append(r); return r
    async def ask(stage, request):
        nonlocal calls
        record({"kind": "request", "stage": stage, "request": request}); calls += 1
        raw = await ask_api(request)
        public = {k: raw.get(k) for k in ("id", "model", "status", "incomplete_details", "usage")}
        public["output"] = [item for item in raw.get("output", []) if item.get("type") == "message"]
        record({"kind": "response", "stage": stage, "response": public})
        parsed = delivered_text(public)
        if parsed["native_refusal"]: raise RuntimeError("native refusal is not a negative label")
        return parsed["text"]
    try:
        write(directory / "INITIAL.json", {"case": case, "phase": "before_planner", "at_utc": now()})
        planner = await ask("planner", study.request(case, "planner")); study.parse_planner(planner)
        pre = record({"kind": "forecast_prefix", "value": study.prefix(case, planner)})
        write(directory / "PREFIX.json", pre)
        await forecast(out, directory, plan)
        record({"kind": "frozen_forecast", "file": "FORECAST.json", "sha256": sha(directory / "FORECAST.json")})
        prediction_made = True
        engine = study.SimulatedDispatchSlots(case); dispatches = []
        for branch in case["branches"]:
            actor = branch["actor"]; text = await ask(actor, study.request(case, actor, planner))
            command = study.parse_buyer(case, actor, text)
            record({"kind": "parsed_proposal", "actor": actor, "command": command})
            event = engine.execute(actor, command)
            record({"kind": "runtime_event", "event": event}); dispatches.append(event)
        write(directory / "OUTCOME.json", study.outcome(case, dispatches)); complete = True
    except Exception as exc:
        failure = type(exc).__name__
        record({"kind": "failure", "error_type": failure, "http_status": getattr(exc, "status_code", None)})
    report = {"case_id": case["id"], "completed_at_utc": now(), "complete": complete, "api_calls": calls,
              "forecast_made": prediction_made, "failure_type": failure,
              "files_sha256": {p.name: sha(p) for p in sorted(directory.glob("*.json"))}}
    write(directory / "REPORT.json", report)
    print(json.dumps({"flow": case["id"], "complete": complete, "api_calls": calls, "forecast": prediction_made, "failure": failure}), flush=True)
    return report


async def collect(out):
    plan = load(out)
    from openai import AsyncOpenAI
    key = os.environ.get("PAPERALPHA_LOCAL_API_KEY")
    if not key: raise RuntimeError("PAPERALPHA_LOCAL_API_KEY is required; never store it in artifacts")
    api = AsyncOpenAI(base_url=plan["api"]["endpoint"], api_key=key, timeout=plan["api"]["timeout_seconds"], max_retries=0)
    try:
        listed = await api.models.list(); names = [m.id for m in listed.data]
        if plan["api"]["model"] not in names: raise RuntimeError("requested model not listed by local endpoint")
        write(out / "API_PREFLIGHT.json", {"at_utc": now(), "authenticated": True, "models": names})
    except Exception as exc:
        write(out / "API_PREFLIGHT_FAILURE.json", {"at_utc": now(), "type": type(exc).__name__})
        await api.close(); raise RuntimeError("local endpoint preflight failed; no generation attempted") from None
    async def ask(request): return (await api.responses.create(**request)).model_dump(mode="json")
    semaphore = asyncio.Semaphore(plan["api"]["concurrent_flows"])
    async def task(case):
        async with semaphore: return await run_flow(case, out, plan, ask)
    try:
        reports = await asyncio.gather(*(task(case) for case in plan["cases"]))
        summary = {"plan_sha256": sha(out / "PLAN.json"), "completed_at_utc": now(), "planned_flows": len(plan["cases"]),
                   "complete_flows": sum(r["complete"] for r in reports), "forecasted_flows": sum(r["forecast_made"] for r in reports),
                   "api_calls": sum(r["api_calls"] for r in reports),
                   "reports_sha256": {r["case_id"]: sha(out / "runs" / r["case_id"] / "REPORT.json") for r in reports},
                   "new_fits": 0, "new_default_selected": False}
        write(out / "COLLECTION.json", summary); print(json.dumps(summary), flush=True)
    finally: await api.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("prepare", "collect")); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-source", type=Path); args = p.parse_args()
    if args.mode == "prepare": prepare(args.out.resolve(), args.model_source.resolve())
    else: asyncio.run(collect(args.out.resolve()))
