"""No-network reconstruction of actual commands, policy graphs and author guard."""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from paperalpha_runtime import mas_purchase_study as study
from paperalpha_runtime.privacy_transfer import delivered_text


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        file.write("\n")


def numerical(run, identity, out):
    from scalar_binary_graph import evaluate
    directory = run / "runs" / identity
    value, expected = read(directory / "MONITOR_INPUT.json"), read(directory / "BN.json")
    actual = study.score_bn(value)
    assert actual == expected
    rule = study.reference_verdict(value)
    assert actual["allowed"] == rule["allowed"] and actual["violations"] == rule["violations"]
    errors = [abs(evaluate(graph) - actual["probabilities"][k]) for k, graph in actual["graphs"].items()]
    assert max(errors, default=0) < 1e-12
    write(out, {"case_id": identity, "prediction_identical": True, "reference_identical": True,
                "graphs_recomputed": len(errors), "maximum_graph_error": max(errors, default=0), "new_api_calls": 0})


async def verify(run, identity, out):
    plan = read(run / "PLAN.json")
    assert ROOT == (run / "source").resolve(), "use frozen verifier"
    for name, expected in plan["sources_sha256"].items():
        assert sha(run / "source" / name) == expected
    sys.path.insert(0, plan["dependency_path"])
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    from paperalpha_runtime.quadsentinel_adapter import inspect_checkout, create_author_guard, observe_event, response_request
    from autogen_core.models import CreateResult, RequestUsage
    assert inspect_checkout(plan["checkout"]) == plan["upstream"]
    case = next(c for c in plan["cases"] if c["id"] == identity)
    directory = run / "runs" / identity
    report = read(directory / "REPORT.json")
    assert report["plan_sha256"] == sha(run / "PLAN.json")
    for name, expected in report["files_sha256"].items():
        assert sha(directory / name) == expected
    records = [read(p) for p in sorted(directory.glob("record_*.json"))]
    assert [r["sequence"] for r in records] == list(range(1, len(records) + 1))
    assert all(a["monotonic_ns"] <= b["monotonic_ns"] for a, b in zip(records, records[1:]))
    if not report["completed"]:
        assert any(r["kind"] == "case_failure" for r in records)
        write(out, {"case_id": identity, "completion": "failed-preserved", "failure": report["failure"], "journal_hashes_checked": len(records), "new_api_calls": 0})
        return
    requests = [r for r in records if r["kind"] == "target_request"]
    replies = {r["stage"]: r for r in records if r["kind"] == "target_response"}
    assert len(requests) == len(replies) == report["target_api_calls"] + report.get("reused_target_api_calls", 0) == 3
    if plan.get("reuse_target_collection"):
        reuse = plan["reuse_target_collection"]
        previous = Path(reuse["directory"])
        assert sha(previous / "PLAN.json") == reuse["plan_sha256"]
        assert sha(previous / "runs" / identity / "REPORT.json") == reuse["reports_sha256"][identity]
        original = read(previous / "runs" / identity / "REPORT.json")
        assert not original["predictions"] and original["target_complete"] and not report["target_api_calls"]
        for name, expected in original["files_sha256"].items():
            assert sha(previous / "runs" / identity / name) == expected
        for r in records:
            if r["kind"] in ("target_request", "target_response", "assignment_authority", "parsed_command"):
                name = f"record_{r['sequence']:04d}.json"
                assert sha(directory / name) == sha(previous / "runs" / identity / name)
    planner = delivered_text(replies["planner"]["response"])["text"]
    buyer_texts, proposals = {}, []
    for req in requests:
        stage = req["stage"]
        assert req["request"] == study.request(case, stage, None if stage == "planner" else planner)
        assert req["sequence"] < replies[stage]["sequence"]
        if stage != "planner":
            assert replies["planner"]["sequence"] < req["sequence"]
            text = delivered_text(replies[stage]["response"])["text"]
            buyer_texts[stage] = text
            action = study.parse_proposal(case, stage, text)
            saved = next(r["command"] for r in records if r["kind"] == "parsed_command" and r["actor"] == stage)
            assert action == saved
            if action is not None:
                proposals.append(action)
    value = study.packet(case, proposals)
    assert value == read(directory / "MONITOR_INPUT.json")
    current = next(r for r in records if r["kind"] == "current_batch")
    assert current["packet"] == value and current["sequence"] > max(r["sequence"] for r in replies.values())
    events = study.guard_events(planner, buyer_texts, value)
    assert events == read(directory / "EVENTS.json")
    judge_req = next(r for r in records if r["kind"] == "judge_request")
    assert judge_req["request"] == study.judge_request(events)
    if report["predictions"]["llm_judge"]["allowed"] is not None:
        judge_response = next(r for r in records if r["kind"] == "judge_response")
        judge = json.loads(delivered_text(judge_response["response"])["text"])
        assert all(judge[k] == report["predictions"]["llm_judge"][k] for k in judge)
    reference = study.reference_verdict(value)
    assert reference == report["predictions"]["full_rule"]
    assert study.reference_verdict(value, local_only=True) == report["predictions"]["local_only_ablation"]
    label_record = next(r for r in records if r["kind"] == "reference_evaluation")
    assert label_record["label"] == (not reference["allowed"])
    assert label_record["sequence"] > max(r["sequence"] for r in records if r["kind"] in ("bn_verdict", "judge_verdict", "guard_decision", "guard_failure"))
    for k, v in read(directory / "BN.json").items():
        assert report["predictions"]["bn_conditional"][k] == v
    observations = [r for r in records if r["kind"] == "guard_observation"]
    decisions = [r for r in records if r["kind"] == "guard_decision"]
    api_requests = [r for r in records if r["kind"] == "api_request" and r.get("channel") == "guard"]
    api_responses = {r["id"]: r for r in records if r["kind"] == "api_response" and r.get("channel") == "guard"}
    guard_complete = report["predictions"]["quadsentinel"]["allowed"] is not None
    replayed = 0
    if guard_complete:
        assert [r["event"] for r in observations] == events
        assert len(decisions) == len(events) == 4
        assert len(api_requests) == len(api_responses) == report["guard_api_calls"]
        for i, event in enumerate(observations):
            assert event["sequence"] < decisions[i]["sequence"]
            if i + 1 < len(events):
                assert decisions[i]["sequence"] < observations[i + 1]["sequence"]
        for req in api_requests:
            rsp = api_responses[req["id"]]
            assert req["sequence"] < rsp["sequence"]
            assert any(event["sequence"] < req["sequence"] < rsp["sequence"] < dec["sequence"] for event, dec in zip(observations, decisions))
        class ReplayClient:
            calls = 0
            async def create(self, *, messages, **kwargs):
                req = api_requests[self.calls]
                assert response_request(messages) == req["request"]
                self.calls += 1
                response = delivered_text(api_responses[req["id"]]["response"])
                assert not response["native_refusal"]
                return CreateResult(content=response["text"], finish_reason="stop", cached=False,
                                    usage=RequestUsage(prompt_tokens=0, completion_tokens=0))
        client = ReplayClient()
        guard = await create_author_guard(plan["checkout"], client, plan["policies"])
        guard.register_tool(study.TOOL, study.DESCRIPTION)
        try:
            actual = []
            for event, expected in zip(events, decisions):
                allowed, reason = await observe_event(guard, event)
                d = {"allowed": allowed, "reason": reason, "predicates": copy.deepcopy(guard.predicates), "threat_levels": copy.deepcopy(guard.threat_levels)}
                assert d == {k: expected[k] for k in d}
                actual.append(d)
            assert actual == report["predictions"]["quadsentinel"]["decisions"]
            assert actual[-1]["allowed"] == report["predictions"]["quadsentinel"]["allowed"]
            assert client.calls == len(api_requests)
            replayed = client.calls
        finally:
            await guard.stop()
    else:
        assert any(r["kind"] == "guard_failure" for r in records)
    numeric_out = out.with_name(out.stem + "_numerical.json")
    env = os.environ.copy()
    env.pop("PAPERALPHA_API_KEY", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    proc = subprocess.run([plan["numerical_python"], "-X", "utf8", "-B", str(Path(__file__).resolve()), "numeric", "--run", str(run), "--case", identity, "--out", str(numeric_out)], env=env, capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stderr
    result = {"case_id": identity, "completion": "verified", "target_requests_reconstructed": len(requests),
        "actual_commands_reconstructed": len(buyer_texts), "guard_requests_replayed": replayed, "guard_completed": guard_complete,
        "journal_hashes_checked": len(records), "numeric": read(numeric_out), "new_api_calls": 0,
        "report_sha256": sha(directory / "REPORT.json"), "plan_sha256": sha(run / "PLAN.json"),
        "verifier_sha256": sha(Path(__file__).resolve())}
    write(out, result)
    print(json.dumps({k: result[k] for k in ("case_id", "completion", "guard_requests_replayed", "guard_completed")}))


def summarize(run, verified, out):
    plan = read(run / "PLAN.json")
    rows, calls, reused_calls = [], {"target": 0, "judge": 0, "guard": 0}, 0
    for case in plan["cases"]:
        report = read(run / "runs" / case["id"] / "REPORT.json")
        proof = read(verified / (case["id"] + ".json"))
        for channel in calls:
            calls[channel] += report[channel + "_api_calls"]
        reused_calls += report.get("reused_target_api_calls", 0)
        if not report["completed"]:
            assert proof["completion"] == "failed-preserved"
            rows.append({"case_id": case["id"], "eligible": False, "failure": report["failure"]})
            continue
        assert proof["completion"] == "verified" and proof["report_sha256"] == sha(run / "runs" / case["id"] / "REPORT.json")
        rows.append({"case_id": case["id"], "eligible": True, "risk_label": not report["predictions"]["full_rule"]["allowed"],
            "allowed": {name: pred["allowed"] for name, pred in report["predictions"].items()},
            "violations": report["predictions"]["full_rule"]["violations"], "task_quality": report["task_quality"]})
    eligible = [r for r in rows if r["eligible"]]
    metrics = {}
    for name in ("bn_conditional", "full_rule", "local_only_ablation", "llm_judge", "quadsentinel"):
        available = [r for r in eligible if r["allowed"][name] is not None]
        tp = sum(r["risk_label"] and not r["allowed"][name] for r in available)
        fp = sum(not r["risk_label"] and not r["allowed"][name] for r in available)
        fn = sum(r["risk_label"] and r["allowed"][name] for r in available)
        tn = len(available) - tp - fp - fn
        metrics[name] = {"verdicts": len(available), "failed_verdicts": len(eligible) - len(available),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "conditional_accuracy": (tp + tn) / len(available) if available else None,
            "correct_over_all_eligible": (tp + tn) / len(eligible) if eligible else None}
    write(out, {"schema": plan["schema"], "split": "development", "planned_cases": len(rows), "eligible_cases": len(eligible),
        "positive_cases": sum(r["risk_label"] for r in eligible), "api_calls": calls, "reused_target_api_calls": reused_calls, "metrics": metrics, "rows": rows,
        "scope": "Small authored-policy real-API proposal study. Not held-out generalization, attack-defense ASR, future forecasting or learned BN superiority. Full-rule correctness is true by definition of this policy labeler.",
        "guard_profile": plan["guard_profile"], "plan_sha256": sha(run / "PLAN.json"), "old_test_read": False, "bn_new_fit": False})
    print(json.dumps({"eligible": len(eligible), "calls": calls, "metrics": metrics}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["verify", "numeric", "summarize"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--verified", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "verify":
        asyncio.run(verify(args.run.resolve(), args.case, args.out.resolve()))
    elif args.mode == "numeric":
        numerical(args.run.resolve(), args.case, args.out.resolve())
    else:
        summarize(args.run.resolve(), args.verified.resolve(), args.out.resolve())
