"""Verify frozen development acquisition, branch isolation and held-out absence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def visible_cues(case, records):
    grants = {r["payload"]["grant_id"]: r["payload"] for r in records if r["kind"] == "grant"}
    revoked = {r["payload"]["grant_id"] for r in records if r["kind"] == "revoke"}
    result = []
    for field in case["tracked_fields"]:
        child = grants.get(field["grant_id"])
        root = grants.get(child["parent_id"]) if child else None
        result.append("unresolved" if not child or not root else "contradicted" if child["grant_id"] in revoked or root["grant_id"] in revoked else "supported")
    return result


def verify(run, out):
    if out.exists():
        raise ValueError("new audit output required")
    sys.path[:0] = [str(run / "source_snapshot/code/commands"), str(run / "source_snapshot/code/src")]
    from collect_branched_publisher_data import load, development_cases, sha, write
    from paperalpha_runtime.privacy_transfer import delivered_text
    import paperalpha_runtime.branched_publisher_data as runtime
    assert Path(runtime.__file__).is_relative_to(run / "source_snapshot")
    plan = load(run)
    dev = run / "development"
    collection = json.loads((dev / "COLLECTION.json").read_text(encoding="utf-8"))
    assert not collection["test_generated"] and collection["plan_sha256"] == plan["sha256"]
    for name, expected in collection["artifact_sha256"].items():
        assert sha(dev / name) == expected, name
    events = [json.loads(line) for line in (dev / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert all(a["monotonic_ns"] <= b["monotonic_ns"] for a, b in zip(events, events[1:]))
    assert events[0]["kind"] == "start" and events[-1]["kind"] == "complete"
    assert plan["created_at"] <= events[0]["time"] and events[0]["plan_sha256"] == plan["sha256"]
    cases = development_cases(plan)
    case_ids = {c["id"] for c in cases}
    records = {r["id"]: r for r in collection["cases"]}
    assert set(records) == case_ids and len(records) == len(collection["cases"])
    assert all(e["case_id"] is None or e["case_id"] in case_ids for e in events)
    test_ids = {c["id"] for c in plan["cases"] if c["split"] == "test"}
    assert not case_ids & test_ids
    split_templates = [{c["template_id"] for c in plan["cases"] if c["split"] == s} for s in ("train", "validation", "test")]
    assert all(not split_templates[i] & split_templates[j] for i in range(3) for j in range(i + 1, 3))
    models, profiles, cue_outcomes, condition_counts = Counter(), Counter(), Counter(), {}
    complete, failures, overlapping_workers = [], [], 0
    active_requests, max_active = set(), 0
    for event in events:
        if event["kind"] == "api_request":
            active_requests.add(event["seq"])
            max_active = max(max_active, len(active_requests))
        elif event["kind"] in {"api_response", "call_failure"}:
            active_requests.discard(event["request_seq"])
    assert not active_requests and max_active <= plan["settings"]["api_concurrency"]
    for case in cases:
        record = records[case["id"]]
        assert record == json.loads((dev / "cases" / (case["id"] + ".json")).read_text(encoding="utf-8"))
        assert record["split"] == case["split"] != "test"
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
            models[str(response["response"].get("model"))] += 1
            try:
                visible = delivered_text(response["response"])
            except ValueError:
                assert any(e["kind"] == "call_failure" and e["stage"] == stage for e in local)
                continue
            texts[stage] = visible["text"]
            if visible["native_refusal"]:
                refusals.append(stage)
        for stage, request in req.items():
            kwargs = {} if stage == "planner" else {"workers": {w: texts[w] for w in ("worker-a", "worker-b")}} if stage == "publisher" else {"planner_text": texts["planner"]}
            expected = {**runtime.stage_request(case, stage, **kwargs), **{k: plan["settings"][k] for k in ("model", "reasoning", "max_output_tokens", "store")}}
            assert request["request"] == expected
            if stage.startswith("worker-"):
                assert resp["planner"]["seq"] < request["seq"]
                payload = json.loads(request["request"]["input"][0]["content"])
                assert "other_worker_message" not in payload and payload["planner_message"] == texts["planner"]
        expected_calls = sorted([{"stage": s, "request_seq": req[s]["seq"], "response_seq": r["seq"]} for s, r in resp.items()], key=lambda r: r["request_seq"])
        assert expected_calls == record["calls"] and set(record["native_refusal_stages"]) == set(refusals)
        if all(w in resp for w in ("worker-a", "worker-b")):
            overlapping_workers += int(max(req[w]["seq"] for w in ("worker-a", "worker-b")) < min(resp[w]["seq"] for w in ("worker-a", "worker-b")))
        prefix_events = [e for e in local if e["kind"] == "prefix_ready"]
        if prefix_events:
            assert len(prefix_events) == 1
            ready = prefix_events[0]
            transition = next(e for e in local if e["kind"] == "authority_transition")
            assert transition["records"] == runtime.transitions(case)
            assert max(resp[w]["seq"] for w in ("worker-a", "worker-b")) < transition["seq"] < ready["seq"] < req["publisher"]["seq"]
            assert ready["seq"] == record["prefix_seq"]
            workers = {w: texts[w] for w in ("worker-a", "worker-b")}
            actual = runtime.issuance(case) + transition["records"]
            assert ready["observation"] == runtime.prefix(case, texts["planner"], workers, actual, runtime.stage_request(case, "publisher", workers=workers))
        if record["status"] == "completed":
            assert set(texts) == set(runtime.STAGES) and len(prefix_events) == 1
            label = runtime.outcome(case, actual, texts["publisher"])
            final = next(e for e in local if e["kind"] == "outcome")
            assert final["seq"] == record["outcome_seq"] > resp["publisher"]["seq"]
            assert final["outcome"] == label
            profile = runtime.literal_profile(case, texts["planner"], workers, texts["publisher"])
            assert final["literal_profile"] == profile
            cues = visible_cues(case, ready["observation"]["publisher_payload"]["cached_authority_records"])
            for item, cue in zip(profile, cues):
                profiles[f"C{item['planner']}W{item['workers'][item['owner']]}F{item['final']}"] += 1
                cue_outcomes[f"{cue}:F{item['final']}"] += 1
            counts = condition_counts.setdefault(case["condition"], {"completed": 0, "at_least_one": 0, "at_least_two": 0, "final_present": 0, "tracked_values": 0})
            counts["completed"] += 1
            counts["at_least_one"] += label["targets"]["1"]
            counts["at_least_two"] += label["targets"]["2"]
            counts["final_present"] += sum(label["appearance"])
            counts["tracked_values"] += len(label["appearance"])
            complete.append({"id": case["id"], "split": case["split"], "template_id": case["template_id"], "prefix": ready,
                "outcome": label, "literal_profile": profile, "visible_cues": cues})
        else:
            assert record["status"] == "failed" and not any(e["kind"] == "outcome" for e in local)
            assert sorted(record["failure_sequences"]) == sorted(e["seq"] for e in local if e["kind"] == "call_failure")
            failures.append(record)
    requests = sum(e["kind"] == "api_request" for e in events)
    assert requests <= plan["maximum_development_calls"]
    report = {"plan_sha256": plan["sha256"], "collection_sha256": sha(dev / "COLLECTION.json"), "artifact_hashes_checked": len(collection["artifact_sha256"]),
        "completed": len(complete), "failed": len(failures), "api_requests": requests, "api_responses": sum(models.values()),
        "response_reported_models": dict(models), "max_concurrent_api_requests": max_active, "runs_with_worker_overlap": overlapping_workers,
        "committed_prefixes": sum(e["kind"] == "prefix_ready" for e in events), "prefixes_before_publisher_calls": True,
        "field_profiles": dict(profiles), "visible_cue_and_retention": dict(cue_outcomes), "by_condition": condition_counts,
        "split_completed": {s: sum(r["split"] == s for r in complete) for s in ("train", "validation")},
        "held_out_configured_runs": len(test_ids), "held_out_generated_runs": 0, "failures": failures,
        "scope": "Training/validation acquisition and coverage only; no monitor scores, fitted improvement or held-out evaluation"}
    write(out / "REPORT.json", report)
    for split in ("train", "validation"):
        write(out / (split + ".json"), [r for r in complete if r["split"] == split])
    print(json.dumps({k: report[k] for k in ("completed", "failed", "api_requests", "field_profiles", "visible_cue_and_retention", "held_out_generated_runs")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
