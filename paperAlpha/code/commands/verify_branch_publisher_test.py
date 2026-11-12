"""Rebuild held-out prefixes/labels, rescore all models, audit pre-dispatch timing."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


def verify(run, out):
    if out.exists():
        raise ValueError("new heldout audit directory required")
    sys.path[:0] = [str(run / "source_snapshot/code/commands"), str(run / "source_snapshot/code/src")]
    from collect_branch_publisher_test import load, sha, write
    from monitor_branch_publisher import FrozenBranchPilot
    from paperalpha_monitor.branch_benchmark import sample_from_prefix, metrics
    from paperalpha_monitor.branch_publisher_bn import BranchPublisherPrivacyBN
    from paperalpha_runtime.privacy_transfer import delivered_text
    from scalar_binary_graph import evaluate
    import paperalpha_runtime.branched_publisher_data as runtime
    assert Path(runtime.__file__).resolve() == run / "source_snapshot/code/src/paperalpha_runtime/branched_publisher_data.py"
    plan = load(run)
    folder = run / "test"
    collection = json.loads((folder / "COLLECTION.json").read_text(encoding="utf-8"))
    assert collection["plan_sha256"] == plan["sha256"]
    for name, expected in collection["artifact_sha256"].items():
        assert sha(folder / name) == expected, name
    events = [json.loads(line) for line in (folder / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert all(a["monotonic_ns"] <= b["monotonic_ns"] for a, b in zip(events, events[1:]))
    assert events[0]["kind"] == "start" and events[-1]["kind"] == "complete"
    assert events[0]["plan_sha256"] == plan["sha256"] and plan["created_at"] <= events[0]["time"]
    original = json.loads((run / "original_task_plan.json").read_text(encoding="utf-8"))
    frozen = json.loads((run / "bundle/FROZEN.json").read_text(encoding="utf-8"))
    assert original["created_at"] <= frozen["created_at"] <= plan["created_at"]
    ids = {c["id"] for c in plan["cases"]}
    assert len(ids) == len(plan["cases"]) == 8
    assert all(e["case_id"] is None or e["case_id"] in ids for e in events)
    records = {r["id"]: r for r in collection["cases"]}
    assert set(records) == ids and len(collection["cases"]) == len(ids)
    active, responses_seen, max_active = set(), set(), 0
    for event in events:
        if event["kind"] == "api_request":
            active.add(event["seq"])
            max_active = max(max_active, len(active))
        elif event["kind"] == "api_response":
            assert event["request_seq"] in active
            active.remove(event["request_seq"])
            responses_seen.add(event["request_seq"])
        elif event["kind"] == "call_failure":
            assert event["request_seq"] in active or event["request_seq"] in responses_seen
            active.discard(event["request_seq"])
    assert not active and max_active <= plan["settings"]["api_concurrency"]
    monitor = FrozenBranchPilot(run / "bundle")
    complete, failures, graph_errors, replays = [], [], [], []
    model_counts, profiles, cues_and_final, by_condition = Counter(), Counter(), Counter(), {}
    all_prediction_count, overlapping_workers = 0, 0
    for case in plan["cases"]:
        record = records[case["id"]]
        assert record == json.loads((folder / "cases" / (case["id"] + ".json")).read_text(encoding="utf-8"))
        assert record["split"] == case["split"] == "test"
        local = [e for e in events if e["case_id"] == case["id"]]
        assert local[0]["kind"] == "authority_initialized" and local[0]["records"] == runtime.issuance(case)
        assert local[-1]["kind"] == "case_complete" and local[-1]["status"] == record["status"]
        req = {e["stage"]: e for e in local if e["kind"] == "api_request"}
        resp = {e["stage"]: e for e in local if e["kind"] == "api_response"}
        assert len(req) == sum(e["kind"] == "api_request" for e in local) <= 4
        assert len(resp) == sum(e["kind"] == "api_response" for e in local)
        texts, refusals = {}, []
        for stage, response in resp.items():
            assert response["request_seq"] == req[stage]["seq"] < response["seq"]
            model_counts[str(response["response"].get("model"))] += 1
            try:
                visible = delivered_text(response["response"])
            except ValueError:
                assert any(e["kind"] == "call_failure" and e["stage"] == stage for e in local)
                continue
            texts[stage] = visible["text"]
            if visible["native_refusal"]:
                refusals.append(stage)
        for stage, sent in req.items():
            kwargs = {} if stage == "planner" else {"workers": {w: texts[w] for w in ("worker-a", "worker-b")}} if stage == "publisher" else {"planner_text": texts["planner"]}
            expected = {**runtime.stage_request(case, stage, **kwargs), **{k: plan["settings"][k] for k in ("model", "reasoning", "max_output_tokens", "store")}}
            assert sent["request"] == expected
            if stage.startswith("worker-"):
                assert resp["planner"]["seq"] < sent["seq"]
        assert record["calls"] == sorted([{"stage": s, "request_seq": req[s]["seq"], "response_seq": r["seq"]} for s, r in resp.items()], key=lambda r: r["request_seq"])
        assert set(record["native_refusal_stages"]) == set(refusals)
        assert record["failures"] == sorted(e["seq"] for e in local if e["kind"] in {"call_failure", "monitor_failure"})
        if all(w in resp for w in ("worker-a", "worker-b")):
            overlapping_workers += int(max(req[w]["seq"] for w in ("worker-a", "worker-b")) < min(resp[w]["seq"] for w in ("worker-a", "worker-b")))
        ready_events = [e for e in local if e["kind"] == "prefix_ready"]
        predictions = [e for e in local if e["kind"] == "prediction"]
        assert len(ready_events) <= 1 and len(predictions) <= 1
        if ready_events:
            ready = ready_events[0]
            updates = [e for e in local if e["kind"] == "authority_transition"]
            assert len(updates) == 1 and updates[0]["records"] == runtime.transitions(case)
            assert max(resp[w]["seq"] for w in ("worker-a", "worker-b")) < updates[0]["seq"] < ready["seq"] == record["prefix_seq"]
            workers = {w: texts[w] for w in ("worker-a", "worker-b")}
            actual = runtime.issuance(case) + updates[0]["records"]
            assert ready["observation"] == runtime.prefix(case, texts["planner"], workers, actual, runtime.stage_request(case, "publisher", workers=workers))
        if predictions:
            assert ready_events
            prediction = predictions[0]
            assert prediction["prefix_seq"] == ready["seq"] < prediction["seq"] == record["prediction_seq"] < req["publisher"]["seq"]
            replay = monitor.score({"observation": ready["observation"], "prefix_sequence": ready["seq"]})
            assert replay == prediction["result"]
            sample = sample_from_prefix(ready)
            for family, model in monitor.models.items():
                if not hasattr(model, "probability_graph"):
                    continue
                bound = sample if isinstance(model, BranchPublisherPrivacyBN) else sample.release
                for k in (1, 2):
                    graph_errors.append(abs(evaluate(model.probability_graph(bound, at_least=k)) - replay["scores"][str(k)][family]))
            all_prediction_count += len(replay["selected"])
            replays.append({"id": case["id"], "prefix_sequence": ready["seq"], "prediction_sequence": prediction["seq"], "publisher_request_sequence": req["publisher"]["seq"], "result": replay})
        else:
            assert "publisher" not in req
        if record["status"] == "completed":
            assert set(texts) == set(runtime.STAGES) and predictions and not record["failures"]
            labels = runtime.outcome(case, actual, texts["publisher"])
            final = [e for e in local if e["kind"] == "outcome"]
            assert len(final) == 1 and final[0]["seq"] == record["outcome_seq"] > resp["publisher"]["seq"]
            assert final[0]["outcome"] == labels
            profile = runtime.literal_profile(case, texts["planner"], workers, texts["publisher"])
            assert final[0]["literal_profile"] == profile
            for j, item in enumerate(profile):
                profiles[f"C{item['planner']}W{item['workers'][item['owner']]}F{item['final']}"] += 1
                cue = sample.release.visible_features[j]
                name = "supported" if cue[0] else "contradicted" if cue[1] else "unresolved"
                cues_and_final[name + ":F" + str(item["final"])] += 1
            counts = by_condition.setdefault(case["condition"], {"completed": 0, "at_least_one": 0, "at_least_two": 0, "final_present": 0})
            counts["completed"] += 1
            counts["at_least_one"] += labels["targets"]["1"]
            counts["at_least_two"] += labels["targets"]["2"]
            counts["final_present"] += sum(labels["appearance"])
            complete.append({"id": case["id"], "template_id": case["template_id"], "outcome": labels,
                "count_distributions": replay["count_distributions"], "profile": profile})
        else:
            assert record["status"] == "failed" and record["failures"] and not any(e["kind"] == "outcome" for e in local)
            failures.append(record)
    scores = {family: metrics([r["count_distributions"][family] for r in complete], [r["outcome"] for r in complete]) for family in monitor.models} if complete else {}
    requests = sum(e["kind"] == "api_request" for e in events)
    assert requests <= plan["maximum_api_calls"] and (not graph_errors or max(graph_errors) < 1e-12)
    assert events[-1]["cases"] == 8 and events[-1]["completed"] == len(complete)
    result = {"plan_sha256": plan["sha256"], "collection_sha256": sha(folder / "COLLECTION.json"),
        "source_sha256": sha(Path(__file__)), "artifact_hashes_checked": len(collection["artifact_sha256"]),
        "planned_cases": 8, "completed": len(complete), "failed": len(failures), "api_requests": requests,
        "api_responses": sum(model_counts.values()), "response_reported_models": dict(model_counts),
        "max_concurrent_api_requests": max_active, "runs_with_worker_overlap": overlapping_workers,
        "prefixes_scored_before_publisher_call": len(replays), "model_predictions_recomputed": all_prediction_count,
        "executable_graphs": len(graph_errors), "max_graph_error": max(graph_errors, default=0.),
        "scores": scores, "field_profiles": dict(profiles), "visible_cue_and_retention": dict(cues_and_final),
        "by_condition": by_condition, "failures": failures, "models_refitted": False, "selected_models_changed": False,
        "scope": "Fixed related-template heldout pilot. Metrics conditional on completed runs; no broad MAS, calibration or original-method superiority claim"}
    write(out / "REPORT.json", result)
    write(out / "recomputed_predictions.json", replays)
    write(out / "completed_cases.json", complete)
    print(json.dumps({k: result[k] for k in ("completed", "failed", "api_requests", "prefixes_scored_before_publisher_call", "scores", "field_profiles")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
