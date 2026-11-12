"""Audit the frozen API pilot, prior-only predictions and independent graphs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def same_json(left, right):
    """JSON stores immutable in-memory tuples as arrays; numbers stay exact."""
    return json.loads(json.dumps(left, allow_nan=False)) == json.loads(json.dumps(right, allow_nan=False))


def metrics(rows, views, priors, targets):
    result = {}
    for view in views:
        result[view] = {}
        for prior in priors:
            key = str(prior)
            result[view][key] = {}
            for target in targets:
                k = str(target)
                y = np.array([r["outcome"]["targets"][k] for r in rows])
                methods = rows[0]["predictions"][view]["results_by_assumed_prior"][key]["scores"][k] if rows else []
                scores = {}
                for method in methods:
                    p = np.array([r["predictions"][view]["results_by_assumed_prior"][key]["scores"][k][method] for r in rows])
                    clipped = np.clip(p, 1e-12, 1 - 1e-12)
                    scores[method] = {"n": len(y), "positives": int(y.sum()), "brier": float(np.mean((p - y) ** 2)),
                        "nll": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
                        "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                        "average_precision": float(average_precision_score(y, p)) if len(set(y)) == 2 else None,
                        "alerts_at_0.5": int(np.sum(p >= .5)), "false_alerts_at_0.5": int(np.sum((p >= .5) & (y == 0))),
                        "misses_at_0.5": int(np.sum((p < .5) & (y == 1)))}
                result[view][key][k] = scores
    return result


def verify(run, out):
    if out.exists():
        raise ValueError("new independent audit directory required")
    sys.path[:0] = [str(run / "source_snapshot/code/src"), str(run / "source_snapshot/code/commands")]
    from collect_scoped_privacy_api import load, sha, write
    from paperalpha_runtime.privacy_transfer import STAGES, delivered_text
    from paperalpha_runtime.scoped_privacy_api import issuance, authority_transition, stage_request, observation, outcome
    from paperalpha_monitor.scoped_privacy_monitor import ScopedPrivacyMonitor, compile_observation
    import paperalpha_monitor.scoped_privacy_monitor as implementation
    from scalar_binary_graph import evaluate
    assert Path(implementation.__file__).is_relative_to(run / "source_snapshot")
    plan = load(run)
    collection = json.loads((run / "COLLECTION.json").read_text(encoding="utf-8"))
    for name, expected in collection["artifacts_sha256"].items():
        assert sha(run / name) == expected, name
    events = [json.loads(line) for line in (run / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert all(a["monotonic_ns"] <= b["monotonic_ns"] for a, b in zip(events, events[1:]))
    assert events[0]["kind"] == "start" and events[-1]["kind"] == "complete"
    assert events[0]["plan_sha256"] == plan["sha256"] == collection["plan_sha256"]
    assert plan["created_at"] <= events[0]["time"]
    monitor = ScopedPrivacyMonitor(run / "bundle")
    records = {r["id"]: r for r in collection["cases"]}
    assert len(records) == len(collection["cases"]) == len(plan["cases"])
    assert set(records) == {c["id"] for c in plan["cases"]}
    assert all(e["case_id"] is None or e["case_id"] in records for e in events)
    complete, failed, errors, profiles, response_models = [], [], [], Counter(), Counter()
    predictions_checked = 0
    for case in plan["cases"]:
        rec = records[case["id"]]
        assert rec == json.loads((run / "cases" / (case["id"] + ".json")).read_text(encoding="utf-8"))
        local = [e for e in events if e["case_id"] == case["id"]]
        assert local[0]["kind"] == "authority_initialized" and local[0]["records"] == issuance(case)
        requests = [e for e in local if e["kind"] == "api_request"]
        assert [e["stage"] for e in requests] == list(STAGES[:len(requests)]) and len(requests) <= 3
        prior, messages, calls = None, [], []
        for index, request in enumerate(requests):
            expected = {**stage_request(case, STAGES[index], prior), **{k: plan["settings"][k] for k in ("model", "reasoning", "max_output_tokens", "store")}}
            assert request["request"] == expected
            responses = [e for e in local if e["kind"] == "api_response" and e["stage"] == STAGES[index]]
            assert len(responses) <= 1
            if not responses:
                assert rec["status"] == "failed" and index == len(requests) - 1
                break
            response = responses[0]
            assert response["request_seq"] == request["seq"] < response["seq"]
            calls.append({"stage": STAGES[index], "request_seq": request["seq"], "response_seq": response["seq"]})
            response_models[str(response["response"].get("model"))] += 1
            try:
                visible = delivered_text(response["response"])
            except ValueError:
                assert rec["status"] == "failed" and index == len(requests) - 1
                break
            messages.append(visible)
            prior = visible["text"]
        assert calls == rec["calls"]
        saved_predictions = [e for e in local if e["kind"] == "prediction"]
        assert [e["seq"] for e in saved_predictions] == rec["predictions"]
        transitions = [e for e in local if e["kind"] == "authority_transition"]
        actual_records = issuance(case)
        if transitions:
            assert len(transitions) == 1 and len(messages) >= 2
            transition = transitions[0]
            assert transition["records"] == authority_transition(case)
            assert transition["stable_release_window"] == {"first_sequence": 6, "last_sequence": 30}
            assert calls[1]["response_seq"] < transition["seq"]
            actual_records += transition["records"]
        predictions = {}
        assert [p["view"] for p in saved_predictions] == plan["views"][:len(saved_predictions)]
        for saved in saved_predictions:
            assert transitions and saved["ok"] and transition["seq"] < saved["seq"]
            if len(requests) == 3:
                assert saved["seq"] < requests[2]["seq"], "prediction after final API request"
            observed = observation(case, messages[0]["text"], messages[1]["text"], actual_records, saved["view"])
            assert saved["observation"] == observed
            replay = monitor.score(observed)
            assert same_json(replay, saved["result"]), "frozen numerical prediction changed"
            sample = compile_observation(observed)
            for p in plan["assumed_priors"]:
                for name, model in monitor.models(p).items():
                    for target in plan["targets"]:
                        independent = evaluate(model.probability_graph(sample, at_least=target))
                        predicted = replay["results_by_assumed_prior"][str(p)]["scores"][str(target)][name]
                        errors.append(abs(independent - predicted))
            predictions[saved["view"]] = replay
            predictions_checked += 1
        if rec["status"] == "completed":
            assert len(messages) == 3 and len(predictions) == 2
            assert [e["kind"] for e in local] == ["authority_initialized", "api_request", "api_response", "api_request", "api_response",
                "authority_transition", "prediction", "prediction", "api_request", "api_response", "outcome"]
            label = outcome(case, actual_records, messages[2]["text"])
            assert local[-1]["outcome"] == label and local[-1]["seq"] == rec["outcome_seq"]
            assert rec["native_refusal_stages"] == [STAGES[i] for i, m in enumerate(messages) if m["native_refusal"]]
            # Independent policy evaluator agrees with full-record graph, for
            # actual literals that appeared and ones that did not.
            assert predictions["full"]["results_by_assumed_prior"]["0.5"]["bn_details"]["bn_coupled"]["reference_failure_marginals"] == label["authority_failure"]
            packet = predictions["full"]["packet"]
            for j, group in enumerate(packet["groups"]):
                profiles[f"C{int(group['coordinator_present'])}W{int(group['worker_present'])}F{label['appearance'][j]}"] += 1
            complete.append({"id": case["id"], "template_id": case["template_id"], "layout": case["layout"], "treatment": case["treatment"],
                "outcome": label, "predictions": predictions})
        else:
            assert rec["status"] == "failed" and local[-1]["kind"] == "failure" and local[-1]["seq"] == rec["failure_seq"]
            assert not any(e["kind"] == "outcome" for e in local)
            failed.append(rec)
    assert max(errors, default=0) < 1e-12
    assert sum(e["kind"] == "api_request" for e in events) <= plan["max_api_calls"]
    report = {"schema": "scoped-privacy-api-audit-v1", "plan_sha256": plan["sha256"], "completed": len(complete), "failed": len(failed),
        "api_requests": sum(e["kind"] == "api_request" for e in events), "api_responses": sum(e["kind"] == "api_response" for e in events),
        "response_reported_models": dict(response_models), "prefixes_recomputed": predictions_checked,
        "executable_graphs_recomputed": len(errors), "max_graph_error": max(errors, default=0.), "profiles_per_tracked_value": dict(profiles),
        "positive_runs_by_target": {str(k): sum(r["outcome"]["targets"][str(k)] for r in complete) for k in plan["targets"]},
        "metrics": metrics(complete, plan["views"], plan["assumed_priors"], plan["targets"]),
        "by_layout": {layout: metrics([r for r in complete if r["layout"] == layout], plan["views"], plan["assumed_priors"], plan["targets"]) for layout in ("shared", "separate")},
        "by_treatment": {t: metrics([r for r in complete if r["treatment"] == t], plan["views"], plan["assumed_priors"], plan["targets"]) for t in ("valid", "stale_root")},
        "predictions_before_final_requests": True, "artifact_hashes_checked": len(collection["artifacts_sha256"]), "failures": failed,
        "scope": plan["protocol"], "independent_templates": len({c["template_id"] for c in plan["cases"]}),
        "numerical_source": str(Path(implementation.__file__).resolve()), "new_fit_or_calibration": False,
        "auditor_sha256": sha(Path(__file__)), "storage_comparison": "Exact JSON-normalized equality; tuple/list representation only, no float tolerance in replay"}
    write(out / "REPORT.json", report)
    write(out / "recomputed_predictions.json", complete)
    shutil.copyfile(Path(__file__), out / "auditor_source.py")
    print(json.dumps({k: report[k] for k in ("completed", "failed", "api_requests", "positive_runs_by_target", "profiles_per_tracked_value", "max_graph_error")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
