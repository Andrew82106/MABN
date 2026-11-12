"""Rebuild live API pilot observations/effects and replay fitted prefix scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from collect_delegation_api_pilot import load
from harnessaudit_cohort import write_new, sha
from monitor_delegation_pilot import make_compiler, restore
from paperalpha_runtime.delegation_pilot import PilotEnvironment
from paperalpha_runtime.stale_authority_pilot import StaleAuthorityEnvironment
from paperalpha_monitor.delegation_evidence import FEATURE_NAMES
from paperalpha_monitor.authority_qualification import QualifiedDelegationStream


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def verify(run, output):
    if output.exists():
        raise ValueError("new verification directory required")
    plan = load(run / "PLAN.json")
    report = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
    assert report["plan_sha256"] == plan["sha256"]
    reported_cases = {c["case_id"]: c for c in report["cases"]}
    assert len(reported_cases) == len(plan["cases"]) == len(report["cases"])
    saved_model, logistic = restore(run / "model.json"), json.loads((run / "logistic.json").read_text(encoding="utf-8"))
    cases, predictions, masks = [], [], []
    maximum_error, checked_files = 0., 0
    for config in plan["cases"]:
        directory = run / "runs" / config["id"]
        manifest = json.loads((directory / "RUN.json").read_text(encoding="utf-8"))
        assert manifest == reported_cases[config["id"]]
        assert manifest["plan_sha256"] == plan["sha256"]
        assert json.loads((directory / "config.json").read_text(encoding="utf-8")) == config
        assert manifest["status"] in {"completed", "failed"}
        for relative, expected in json.loads((directory / "ARTIFACTS.json").read_text(encoding="utf-8")).items():
            assert sha(directory / relative) == expected
            checked_files += 1
        requests, responses = lines(directory / "api_requests.jsonl"), lines(directory / "api_responses.jsonl")
        assert len(requests) == manifest["API_calls"] and len(responses) == manifest["API_finished"] + manifest["API_failed"]
        for request in requests:
            assert request["model_requested"] == "gpt-5.5"
            assert request["model_settings"]["reasoning"]["effort"] == "low"
        generated_calls = {}
        for response in responses:
            for item in response.get("response", {}).get("output", []):
                if item["type"] == "function_call":
                    assert item["call_id"] not in generated_calls
                    generated_calls[item["call_id"]] = item
        tools, stages = lines(directory / "tool_calls.jsonl"), lines(directory / "stages.jsonl")
        completed = {r["actor"]: r["output"] for r in stages if r["phase"] == "completed"}
        captures, stream = defaultdict(list), QualifiedDelegationStream(saved_model, make_compiler(config))
        environment = StaleAuthorityEnvironment if "stale_authority_view" in config else PilotEnvironment
        env = environment(config, observe=stream.push, save=lambda name, item: captures[name].append(item))
        after_planner, checked_calls, returned = False, 0, {}
        for entry in tools:
            identity = entry["provider_call_id"]
            if entry["phase"] == "entry":
                actual = generated_calls[identity]
                assert actual["name"] == entry["tool"] and json.loads(actual["arguments"]) == entry["arguments"]
                if entry["actor"] == "worker" and not after_planner:
                    env.after_planner(completed["planner"])
                    after_planner = True
                returned[identity] = env.invoke(entry["actor"], entry["tool"], entry["arguments"], identity)
                checked_calls += 1
            elif entry["phase"] == "return":
                assert returned[identity] == entry["result"]
            else:
                raise ValueError("tool failure requires a separate failure replay, not silent omission")
        if not after_planner and "planner" in completed:
            env.after_planner(completed["planner"])
        original_world = lines(directory / "world_events.jsonl")
        assert [r["record"] for r in captures["world_events"]] == [r["record"] for r in original_world]
        assert [r["visible_to_monitor"] for r in captures["world_events"]] == [r["visible_to_monitor"] for r in original_world]
        assert captures["monitor_events"] == lines(directory / "monitor_events.jsonl")
        live = {r["sequence"]: r for r in lines(directory / "online_predictions.jsonl")}
        effects = lines(directory / "outcomes.jsonl")
        replayed_effects = captures["outcomes"]
        assert len(effects) == len(replayed_effects)
        for old, new in zip(effects, replayed_effects):
            assert {k: v for k, v in old.items() if k != "recorded_monotonic_ns"} == {
                k: v for k, v in new.items() if k != "recorded_monotonic_ns"}
            before = live[old["sequence"]]
            assert before["record_order"] < old["record_order"]
            assert before["recorded_monotonic_ns"] <= old["recorded_monotonic_ns"]
            now = next(r["prediction"]["dispatches"][0] for r in captures["online_predictions"] if r["sequence"] == old["sequence"])
            original = before["prediction"]["dispatches"][0]
            assert original["reference_status"] == now["reference_status"] and original["features"] == now["features"]
            maximum_error = max(maximum_error, abs(original["model_point_probability"] - now["model_point_probability"]))
            x = np.r_[1., [now["features"][name] for name in FEATURE_NAMES]]
            rule = 0. if now["features"]["support_matched"] else (1. if now["features"]["reference_invalid"] and not now["features"]["reference_unresolved"] else .5)
            predictions.append({"case_id": config["id"], "action_id": old["action_id"], "reference_policy_violation": old["reference_policy_violation"],
                "online_bn": original["model_point_probability"], "same_feature_logistic": float(expit(x @ np.asarray(logistic["theta"]))),
                "visible_reference_rule": rule, "qualified_replay": now})
        assert len(effects) == manifest["simulated_payments"]
        assert sum(r["reference_policy_violation"] for r in effects) == manifest["violations"]
        if config.get("stale_authority_view"):
            assert captures["injections"] == lines(directory / "injections.jsonl")
            # Post-hoc omission diagnostic: not a fresh Agent run or new labels.
            masked_config = copy.deepcopy(config)
            masked_config["coverage_complete"] = False
            masked = QualifiedDelegationStream(saved_model, make_compiler(masked_config))
            for record in captures["monitor_events"]:
                if record.get("kind") == "revoke":
                    continue
                value = masked.push(record)
                masks.extend({"case_id": config["id"], **r,
                    "interpretation": "Same actual action with revocation observation removed; post-hoc information-availability check."}
                    for r in value["dispatches"])
        cases.append({"case_id": config["id"], "variant": config["variant"], "stale_view": config.get("stale_authority_view", False),
            "status": manifest["status"], "API_calls": len(requests), "actual_model_tool_calls_verified": checked_calls,
            "monitor_events": len(captures["monitor_events"]), "payments": len(effects), "violations": manifest["violations"],
            "input_tokens": manifest["input_tokens"], "output_tokens": manifest["output_tokens"]})
    assert maximum_error < 1e-12
    assert sum(c["API_calls"] for c in cases) == report["total_API_calls"]
    output.mkdir(parents=True)
    write_new(output / "action_comparison.json", predictions)
    write_new(output / "missing_revocation_diagnostic.json", masks)
    with (output / "verify_source.py").open("xb") as f:
        f.write(Path(__file__).read_bytes())
    # This qualification wrapper was added after acquisition. Freeze it here,
    # separately from the original runtime sources; never imply it ran online.
    qualification_source = Path(__file__).resolve().parents[2] / 'code/src/paperalpha_monitor/authority_qualification.py'
    with (output / "qualification_source.py").open("xb") as f:
        f.write(qualification_source.read_bytes())
    result = {"run_report_sha256": sha(run / "REPORT.json"), "artifacts_checked": checked_files, "cases": cases,
        "actual_API_calls": sum(c["API_calls"] for c in cases), "max_online_vs_replay_difference": maximum_error,
        "qualification_protocol": "Post-acquisition qualified replay; not the original live explanation. Source frozen alongside verifier.",
        "original_scores_changed": False, "claim": "Live integration and evidence availability only; tiny simulated cohort, no fitted-on-cohort or MAS superiority claim.",
        "artifacts_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write_new(output / "REPORT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
