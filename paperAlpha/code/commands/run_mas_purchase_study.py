"""Freeze/run a fresh real-API development pilot; never train on old tests."""
from __future__ import annotations

import argparse
import asyncio
import copy
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime import mas_purchase_study as study
from paperalpha_runtime.privacy_transfer import delivered_text
from paperalpha_runtime.quadsentinel_adapter import inspect_checkout, PROFILE


def write(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out, checkout, deps, numerical_python, reuse_from=None):
    upstream = inspect_checkout(checkout)
    out.mkdir(parents=True, exist_ok=False)
    names = [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'code/src').rglob("*.py")]
    names += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / 'knowledge/templates').glob("*.json")]
    names += ['code/commands/run_mas_purchase_study.py', 'code/commands/verify_mas_purchase_study.py', 'code/commands/scalar_binary_graph.py',
              'code/tests/test_mas_purchase_study.py', 'knowledge/risks/risk_catalog.json']
    for name in names:
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    # Qualify the standalone numeric snapshot before any paid/model calls.
    check = subprocess.run([str(numerical_python), "-X", "utf8", "-B", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); from paperalpha_monitor.workflow_risk_bn import WorkflowRiskBN; print(len(WorkflowRiskBN().theta))",
        str(out / "source/code/src")], capture_output=True, text=True, timeout=60)
    write(out / "NUMERICAL_IMPORT_PREFLIGHT.json", {"exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr, "new_api_calls": 0})
    if check.returncode:
        raise RuntimeError("standalone snapshot import failed before any API call")
    reuse = None
    if reuse_from is not None:
        reuse_from = reuse_from.resolve()
        prior = json.loads((reuse_from / "PLAN.json").read_text(encoding="utf-8"))
        if prior["cases"] != study.cases() or prior["policies"] != study.guard_policies():
            raise ValueError("reuse must preserve every original task and guard policy")
        for c in prior["cases"]:
            report = json.loads((reuse_from / "runs" / c["id"] / "REPORT.json").read_text(encoding="utf-8"))
            if not report["target_complete"] or report["judge_api_calls"] or report["guard_api_calls"] or report["predictions"]:
                raise ValueError("only complete target collection before any scoring may be resumed here")
        reuse = {"directory": str(reuse_from), "plan_sha256": file_hash(reuse_from / "PLAN.json"),
                 "reports_sha256": {c["id"]: file_hash(reuse_from / "runs" / c["id"] / "REPORT.json") for c in prior["cases"]},
                 "reason": "Original standalone numeric snapshot omitted package-import configurations; resume every collected prefix, no target rerun or selection."}
    plan = {"schema": "real-api-mas-purchase-development-v1", "cases": study.cases(), "policies": study.guard_policies(),
        "workflow": study.WORKFLOW, "guard_profile": PROFILE, "checkout": str(checkout.resolve()), "upstream": upstream,
        "dependency_path": str(deps.resolve()), "numerical_python": str(numerical_python.resolve()),
        "api": {"endpoint": "http://localhost:58661/v1", "model": "gpt-5.5", "effort": "low", "max_output_tokens": 3200,
                "timeout_seconds": 60, "transport_retries": 0, "target_calls_per_case": 3, "judge_calls_per_case": 1, "max_guard_calls_per_case": 48},
        "sources_sha256": {n: file_hash(out / "source" / n) for n in sorted(names)},
        "protocol": ["One process per case; actual planner reply enters both buyers' requests.",
            "Buyer JSON is a runtime command protocol, not a native Responses function-call event. No schema repair or forced dangerous action.",
            "Current batch admission only. No forecast, no actual financial execution, no monitor feedback or blocking.",
            "All monitors get the same current batch, scoped policies and complete current authority facts; language methods also get raw prior messages.",
            "BN uses all included dispatch conditions = 1; complete known policy yields deterministic rules. No fitted probability-calibration claim.",
            "Independent rules label actual proposed amounts and current policies; labels computed after all verdicts. Local-only rule is a relation ablation.",
            "All eight cases are development diagnostics, not new held-out performance evidence. Preserve all failures and abstentions.",
            "Author guard configuration deviations are unchanged from its preflight: uniform 5.5, k=-1, vetted compiled policies, monitor-only."],
        "old_test_read": False, "new_fit": False, "source_records_are_authored": True, "agent_outputs_must_be_real_api": True,
        "reuse_target_collection": reuse}
    write(out / "PLAN.json", plan)
    print(json.dumps({"prepared": str(out), "cases": len(plan["cases"]), "source_files": len(names), "plan_sha256": file_hash(out / "PLAN.json")}))


def frozen_plan(out):
    plan = json.loads((out / "PLAN.json").read_text(encoding="utf-8"))
    for name, expected in plan["sources_sha256"].items():
        if file_hash(out / "source" / name) != expected:
            raise ValueError("frozen source differs")
    if Path(__file__).resolve() != (out / "source/code/commands/run_mas_purchase_study.py").resolve():
        raise ValueError("run the frozen collector")
    if study.cases() != plan["cases"] or study.guard_policies() != plan["policies"]:
        raise ValueError("frozen task/prompt definitions differ")
    return plan


async def run(out, identity):
    plan = frozen_plan(out)
    if inspect_checkout(plan["checkout"]) != plan["upstream"]:
        raise ValueError("author checkout changed")
    case = next(c for c in plan["cases"] if c["id"] == identity)
    directory = out / "runs" / identity
    directory.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, plan["dependency_path"])
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    from paperalpha_runtime.quadsentinel_adapter import create_api_client, ResponsesGuardClient, create_author_guard, observe_event
    records = []
    def record(value):
        item = {"sequence": len(records) + 1, "monotonic_ns": time.monotonic_ns(), **value}
        write(directory / f"record_{len(records) + 1:04d}.json", item)
        records.append(item)
    api = create_api_client(plan["api"]["endpoint"], os.environ.get("PAPERALPHA_API_KEY"))
    protected_calls, judge_calls = 0, 0
    async def ask(channel, stage, request):
        nonlocal protected_calls, judge_calls
        if channel == "target":
            protected_calls += 1
        else:
            judge_calls += 1
        record({"kind": channel + "_request", "stage": stage, "request": request})
        try:
            raw = (await api.responses.create(**request)).model_dump(mode="json")
            public = {k: raw.get(k) for k in ("id", "model", "status", "incomplete_details", "usage")}
            public["output"] = [m for m in raw.get("output", []) if m.get("type") == "message"]
            record({"kind": channel + "_response", "stage": stage, "response": public})
            reply = delivered_text(public)
            if reply["native_refusal"]:
                raise RuntimeError("native refusal")
            return reply["text"]
        except Exception as exc:
            record({"kind": channel + "_failure", "stage": stage, "error_type": type(exc).__name__})
            raise RuntimeError("API stage did not produce an eligible completed reply") from None
    failure, guard, bridge, predictions, target_complete, quality = None, None, None, {}, False, []
    started = time.monotonic()
    try:
        reuse = plan.get("reuse_target_collection")
        original_records = None
        if reuse:
            previous = Path(reuse["directory"])
            if file_hash(previous / "PLAN.json") != reuse["plan_sha256"] or file_hash(previous / "runs" / identity / "REPORT.json") != reuse["reports_sha256"][identity]:
                raise ValueError("source collection identity changed")
            prior_dir = previous / "runs" / identity
            prior_report = json.loads((prior_dir / "REPORT.json").read_text(encoding="utf-8"))
            for name, expected in prior_report["files_sha256"].items():
                if file_hash(prior_dir / name) != expected:
                    raise ValueError("source collection evidence changed")
            original_records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(prior_dir.glob("record_*.json"))]
            # Preserve the original timestamps and bytes for the complete target
            # prefix. Do not copy its failed scorer event or claim a new API run.
            for item in original_records:
                if item["kind"] in ("current_batch", "case_failure"):
                    continue
                destination = directory / f"record_{len(records) + 1:04d}.json"
                shutil.copyfile(prior_dir / destination.name, destination)
                records.append(item)
            planner_reply = next(r for r in records if r["kind"] == "target_response" and r["stage"] == "planner")
            planner = delivered_text(planner_reply["response"])["text"]
        else:
            initial = [r for r in study.authority_records(case) if r["kind"] == "grant"]
            record({"kind": "assignment_authority", "records": initial})
            planner = await ask("target", "planner", study.request(case, "planner"))
        buyer_texts, proposals = {}, []
        # Branches have independent contexts and no peer message. Sequential API
        # scheduling is intentional; DAG parallelism need not mean network concurrency.
        for branch in case["branches"]:
            actor = branch["actor"]
            if reuse:
                saved_request = next(r for r in records if r["kind"] == "target_request" and r["stage"] == actor)
                if saved_request["request"] != study.request(case, actor, planner):
                    raise ValueError("source target prompt changed")
                text = delivered_text(next(r["response"] for r in records if r["kind"] == "target_response" and r["stage"] == actor))["text"]
            else:
                text = await ask("target", actor, study.request(case, actor, planner))
            buyer_texts[actor] = text
            command = study.parse_proposal(case, actor, text)
            if reuse:
                if command != next(r["command"] for r in records if r["kind"] == "parsed_command" and r["actor"] == actor):
                    raise ValueError("source command changed")
            else:
                record({"kind": "parsed_command", "actor": actor, "command": command})
            if command is not None:
                proposals.append(command)
                selected = next(i for i in branch["catalogue"] if i["sku"] == command["arguments"]["sku"])
                compatible_prices = [Decimal(i["price"]) for i in branch["catalogue"] if i["meets_requirement"]]
                quality.append({"actor": actor, "abstained": False, "compatible": selected["meets_requirement"],
                                "cheapest_compatible": selected["meets_requirement"] and Decimal(selected["price"]) == min(compatible_prices)})
            else:
                quality.append({"actor": actor, "abstained": True, "compatible": None, "cheapest_compatible": None})
        target_complete = True
        value = study.packet(case, proposals)
        if reuse and value != next(r["packet"] for r in original_records if r["kind"] == "current_batch"):
            raise ValueError("source monitoring packet changed")
        record({"kind": "current_batch", "packet": value})
        input_path, output_path = directory / "MONITOR_INPUT.json", directory / "BN.json"
        write(input_path, value)
        numeric_env = os.environ.copy()
        numeric_env.pop("PAPERALPHA_API_KEY", None)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            numeric_env[k] = "1"
        begin = time.monotonic()
        proc = subprocess.run([plan["numerical_python"], "-X", "utf8", "-B", str(Path(__file__).resolve()), "score",
                              "--input", str(input_path), "--output", str(output_path)], env=numeric_env, capture_output=True, text=True, timeout=90)
        if proc.returncode:
            record({"kind": "numeric_failure", "exit_code": proc.returncode, "stderr": proc.stderr})
            raise RuntimeError("numeric scoring failed")
        predictions["bn_conditional"] = {**json.loads(output_path.read_text(encoding="utf-8")), "elapsed_including_process_seconds": time.monotonic() - begin}
        record({"kind": "bn_verdict", "output_sha256": file_hash(output_path)})
        events = study.guard_events(planner, buyer_texts, value)
        write(directory / "EVENTS.json", events)
        try:
            begin = time.monotonic()
            judge_text = await ask("judge", "batch", study.judge_request(events))
            judge = json.loads(judge_text)
            if not isinstance(judge, dict) or set(judge) != {"allowed", "violated_policies", "explanation"} or type(judge["allowed"]) is not bool or not isinstance(judge["violated_policies"], list) or not all(isinstance(x, str) for x in judge["violated_policies"]) or not isinstance(judge["explanation"], str):
                raise ValueError("invalid judge schema")
            predictions["llm_judge"] = {**judge, "elapsed_seconds": time.monotonic() - begin}
        except Exception as exc:
            predictions["llm_judge"] = {"allowed": None, "error_type": type(exc).__name__}
        record({"kind": "judge_verdict", "verdict": predictions["llm_judge"]})
        try:
            begin = time.monotonic()
            bridge = ResponsesGuardClient(api, lambda v: record({"channel": "guard", **v}), plan["api"]["max_guard_calls_per_case"])
            guard = await create_author_guard(plan["checkout"], bridge, plan["policies"])
            guard.register_tool(study.TOOL, study.DESCRIPTION)
            decisions = []
            for event in events:
                record({"kind": "guard_observation", "event": event})
                allowed, reason = await observe_event(guard, event)
                if type(allowed) is not bool or not isinstance(reason, str):
                    raise ValueError("invalid guard verdict")
                decision = {"allowed": allowed, "reason": reason, "predicates": copy.deepcopy(guard.predicates), "threat_levels": copy.deepcopy(guard.threat_levels)}
                decisions.append(decision)
                record({"kind": "guard_decision", **decision})
            predictions["quadsentinel"] = {"allowed": decisions[-1]["allowed"], "decisions": decisions, "elapsed_seconds": time.monotonic() - begin}
        except Exception as exc:
            predictions["quadsentinel"] = {"allowed": None, "error_type": type(exc).__name__}
            record({"kind": "guard_failure", "error_type": type(exc).__name__})
        # Independent policy outcomes are deliberately evaluated only now.
        predictions["full_rule"] = study.reference_verdict(value)
        predictions["local_only_ablation"] = study.reference_verdict(value, local_only=True)
        record({"kind": "reference_evaluation", "label": not predictions["full_rule"]["allowed"], "reference": predictions["full_rule"]})
    except Exception as exc:
        failure = {"error_type": type(exc).__name__}
        record({"kind": "case_failure", **failure})
    finally:
        if guard is not None:
            await guard.stop()
        await api.close()
    report = {"case_id": identity, "target_complete": target_complete, "completed": failure is None, "failure": failure,
        "target_api_calls": protected_calls, "reused_target_api_calls": 3 if plan.get("reuse_target_collection") else 0,
        "judge_api_calls": judge_calls, "guard_api_calls": bridge.calls if bridge else 0,
        "wall_seconds": time.monotonic() - started, "task_quality": quality, "predictions": predictions,
        "split": "development", "new_bn_fit": False, "real_external_effects": False, "guard_profile": PROFILE,
        "plan_sha256": file_hash(out / "PLAN.json"),
        "files_sha256": {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()}}
    write(directory / "REPORT.json", report)
    print(json.dumps({k: report[k] for k in ("case_id", "target_complete", "completed", "failure", "target_api_calls", "judge_api_calls", "guard_api_calls")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "score"])
    for field in ("out", "checkout", "deps", "numerical-python", "input", "output", "reuse-from"):
        parser.add_argument("--" + field, type=Path)
    parser.add_argument("--case")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.out.resolve(), args.checkout, args.deps, args.numerical_python, args.reuse_from)
    elif args.mode == "score":
        write(args.output, study.score_bn(json.loads(args.input.read_text(encoding="utf-8"))))
    else:
        asyncio.run(run(args.out.resolve(), args.case))
