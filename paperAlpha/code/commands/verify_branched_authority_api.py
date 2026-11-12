"""Verify actual API calls, event reconstruction, prefix scores and paired views.

Never re-run an agent. Keep incomplete/refusal runs; report tiny cohort results
descriptively, with the originally frozen endpoint eligibility rule.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from collect_branched_authority_api import load, MODELS, instructions
from harnessaudit_cohort import ROOT, sha, write_new
from paperalpha_runtime.branched_authority_pilot import ACTORS, BranchedAuthorityEnvironment, monitor_configuration, stage_input
from paperalpha_monitor.branched_authority_monitor import BranchedAuthorityMonitor


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def without_clock(row):
    return {k: v for k, v in row.items() if k != "recorded_monotonic_ns"}


def initial_input_matches(value, text):
    # The Agents SDK converts Runner's string to a single user message before
    # the recorded provider boundary. Accept the two explicit representations.
    return value == text or value == [{"role": "user", "content": text}]


def verify(run, out):
    if out.exists():
        raise ValueError("new verification directory required")
    plan = load(run / "PLAN.json")
    report = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
    assert report["plan_sha256"] == plan["sha256"]
    manifests = {c["case_id"]: c for c in report["cases"]}
    assert len(manifests) == len(plan["cases"]) == len(report["cases"])
    models = [json.loads((run / name).read_text(encoding="utf-8")) for name in MODELS]
    cases, endpoints, prefixes = [], [], []
    checked = 0
    for config in plan["cases"]:
        folder = run / "runs" / config["id"]
        read = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
        manifest = read("RUN.json")
        assert manifest == manifests[config["id"]] and manifest["plan_sha256"] == plan["sha256"]
        assert read("config.json") == config and read("monitor_config.json") == monitor_configuration(config)
        for name, expected in read("ARTIFACTS.json").items():
            assert sha(folder / name) == expected
            checked += 1
        requests, responses = lines(folder / "api_requests.jsonl"), lines(folder / "api_responses.jsonl")
        assert len(requests) == manifest["API_calls"]
        assert len(responses) == manifest["API_finished"] + manifest["API_failed"]
        by_call = {r["call"]: r for r in requests}
        generated, outputs = {}, defaultdict(list)
        for response in responses:
            request = by_call[response["call"]]
            assert request["model_requested"] == plan["settings"]["model"]
            assert request["model_settings"]["reasoning"]["effort"] == plan["settings"]["reasoning"]
            actor = request["actor"]
            assert request["system_instructions"] == instructions(config, actor)
            for item in response.get("response", {}).get("output", []):
                if item["type"] == "function_call":
                    assert item["call_id"] not in generated
                    generated[item["call_id"]] = (actor, item)
                elif item["type"] == "message":
                    outputs[actor].append("".join(c.get("text", "") for c in item.get("content", []) if c["type"] == "output_text"))
        tools, stages = lines(folder / "tool_calls.jsonl"), lines(folder / "stages.jsonl")
        completed = {r["actor"]: r["output"] for r in stages if r["phase"] == "completed"}
        for stage in stages:
            actor = stage["actor"]
            if stage["phase"] == "started":
                assert stage["input"] == stage_input(config, actor, completed.get("planner"))
                actor_requests = [r for r in requests if r["actor"] == actor]
                if not actor_requests:
                    assert manifest["status"] == "failed"
                    continue
                first = actor_requests[0]
                assert initial_input_matches(first["input"], stage["input"]) and first["system_instructions"] == stage["instructions"]
            else:
                assert stage["output"] in outputs[actor]
        capture = defaultdict(list)
        monitor = BranchedAuthorityMonitor(monitor_configuration(config), *models)
        env = BranchedAuthorityEnvironment(config, observe=monitor.push, save=lambda name, value: capture[name].append(value))
        tool_entries = [r for r in tools if r["phase"] == "entry"]
        returns = {r["provider_call_id"]: r for r in tools if r["phase"] in {"return", "error"}}
        assert len(returns) == len(tool_entries)
        for actor in ACTORS:
            for entry in (r for r in tool_entries if r["actor"] == actor):
                identity = entry["provider_call_id"]
                actual_actor, actual = generated[identity]
                assert actual_actor == actor and actual["name"] == entry["tool"] and json.loads(actual["arguments"]) == entry["arguments"]
                try:
                    result = env.invoke(actor, entry["tool"], entry["arguments"], identity)
                except Exception as exc:
                    assert returns[identity]["phase"] == "error" and returns[identity]["error_type"] == type(exc).__name__
                else:
                    assert returns[identity]["phase"] == "return" and returns[identity]["result"] == result
            if actor == "planner" and actor in completed:
                env.after_planner(completed[actor])
        assert [r["record"] for r in capture["world_events"]] == [r["record"] for r in lines(folder / "world_events.jsonl")]
        assert capture["monitor_events"] == lines(folder / "monitor_events.jsonl")
        assert capture["injections"] == lines(folder / "injections.jsonl")
        outcomes = lines(folder / "outcomes.jsonl")
        assert [without_clock(r) for r in outcomes] == [without_clock(r) for r in capture["outcomes"]]
        live = lines(folder / "online_predictions.jsonl")
        assert len(live) == len(capture["online_predictions"])
        for old, new in zip(live, capture["online_predictions"]):
            # Exact full-object equality includes paths, feature vectors, shared
            # identities and all scores, not only the final BN point estimate.
            assert without_clock(old) == without_clock(new)
        live_by_sequence = {r["sequence"]: r for r in live}
        for i, outcome in enumerate(outcomes):
            old = live_by_sequence[outcome["sequence"]]
            assert old["record_order"] < outcome["record_order"]
            assert old["recorded_monotonic_ns"] <= outcome["recorded_monotonic_ns"]
            target = int(sum(r["reference_policy_violation"] for r in outcomes[:i + 1]) >= 2)
            for view, value in old["prediction"]["views"].items():
                row = value["dispatches"][0]
                assert row["action_id"] == outcome["action_id"]
                prefixes.append({"case_id": config["id"], "view": view, "sequence": outcome["sequence"],
                    "observed_actions": i + 1, "two_failures_label": target, "prediction": row})
        assert len(outcomes) == manifest["simulated_payments"]
        assert sum(r["reference_policy_violation"] for r in outcomes) == manifest["violations"]
        eligible = manifest["status"] == "completed" and len(outcomes) == 2
        if eligible:
            for view in monitor.bases:
                row = next(r for r in reversed(prefixes) if r["case_id"] == config["id"] and r["view"] == view)
                p = row["prediction"]
                endpoints.append({"case_id": config["id"], "layout": config["layout"], "treatment": config["treatment"],
                    "view": view, "label": row["two_failures_label"],
                    "system_bn": p["model_point_probability"], "same_information_logistic": p["graph_logistic_two_failures"],
                    "reference_joint": p["reference_two_failures"], "independent_action_ablation": p["independent_action_two_failures"],
                    "visible_rule": p["visible_rule_two_failures"], "facts": p["shared_fact_count"],
                    "action_marginals": p["action_marginals"], "graph_features": p["same_information_features"]})
        cases.append({"case_id": config["id"], "layout": config["layout"], "treatment": config["treatment"],
            "status": manifest["status"], "API_calls": len(requests), "actual_tools_verified": len(tool_entries),
            "payments": len(outcomes), "violations": manifest["violations"], "stages_completed": list(completed),
            "endpoint_eligible": eligible, "exclusion_reason": None if eligible else "failed run or other than exactly two observed payments",
            "input_tokens": manifest["input_tokens"], "output_tokens": manifest["output_tokens"]})
    assert sum(c["API_calls"] for c in cases) == report["total_API_calls"]
    descriptive = {}
    for view in ("complete", "root_updates_missing"):
        rows = [r for r in endpoints if r["view"] == view]
        descriptive[view] = {"eligible_runs": len(rows), "positive_endpoints": sum(r["label"] for r in rows),
            "brier": {key: float(np.mean([(r[key] - r["label"]) ** 2 for r in rows])) if rows else None
                for key in ("system_bn", "same_information_logistic", "reference_joint", "independent_action_ablation")},
            "rule_known_count": sum(r["visible_rule"] is not None for r in rows)}
    out.mkdir(parents=True)
    write_new(out / "cases.json", cases)
    write_new(out / "endpoints.json", endpoints)
    write_new(out / "prefixes.json", prefixes)
    with (out / "verify_source.py").open("xb") as handle:
        handle.write(Path(__file__).read_bytes())
    result = {"source_report_sha256": sha(run / "REPORT.json"), "artifacts_checked": checked,
        "API_calls": sum(c["API_calls"] for c in cases), "real_tool_calls_checked": sum(c["actual_tools_verified"] for c in cases),
        "payments": sum(c["payments"] for c in cases), "violations": sum(c["violations"] for c in cases),
        "online_replay_full_object_equality": True, "descriptive_endpoint_scores": descriptive,
        "qualification": "Eight hand-configured development cases; paired views are not independent samples; constructed-frequency-trained estimators uncalibrated. No original MAS baseline or superiority claim.",
        "artifacts_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    write_new(out / "REPORT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
