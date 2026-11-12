"""Compile four approved legacy development candidates, without model scoring."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from audit_agentleak_error_eligibility import audit_trace, sha, write_new
from paperalpha_monitor.agentleak_message_compiler import compile_prefix
from paperalpha_monitor.agentleak_event_projection import later_labels
from paperalpha_monitor.observable_event_learning import validated_labels
from paperalpha_monitor.privacy_flow_data import protected_groups

APPROVED = {"trace_20260131_140035_ba61928f": "fit",
            "trace_20260131_140109_7b720077": "fit",
            "trace_20260131_140013_c4e4afce": "validation",
            "trace_20260131_140050_0887cb4a": "validation"}
REPORT_SHA = "ab848b7db9696d8b07df366a869843c3c919cc0115c675b5f4c6344e11340824"


def main():
    directory = ROOT / 'experiments/privacy_flow/runs/agentleak_error_eligibility_audit_20260930'
    output = directory / "PROJECTION_CHECK.json"
    if output.exists(): raise ValueError("projection check exists; never overwrite")
    report_path, selection_path = directory/"REPORT.json", directory/"SELECTION.json"
    assert sha(report_path) == REPORT_SHA
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert sha(selection_path) == report["selection_sha256"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = {r["id"]: r for r in selection["allowed"] if r["id"] in APPROVED}
    approved_rows = {r["id"]: r for r in report["records"] if r["candidate_complete_mas"]}
    assert set(selected) == set(approved_rows) == set(APPROVED)
    plan_path = ROOT / 'experiments/privacy_flow/runs/privacy_flow_agentleak_20260919/PLAN.json'
    assert sha(plan_path) == report["source_sha256_before"][str(plan_path.relative_to(ROOT))]
    templates = {t["template_id"]: t for t in json.loads(plan_path.read_text(encoding="utf-8"))["templates"]}
    source_names = ['code/commands/check_agentleak_error_candidate_projection.py', 'code/commands/audit_agentleak_error_eligibility.py',
        'code/src/paperalpha_monitor/agentleak_message_compiler.py', 'code/src/paperalpha_monitor/agentleak_event_projection.py',
        'code/src/paperalpha_monitor/message_workflow_compiler.py', 'code/src/paperalpha_monitor/observable_event_dag.py',
        'code/src/paperalpha_monitor/observable_event_learning.py', 'code/src/paperalpha_monitor/privacy_flow_data.py',
        'knowledge/templates/message_workflow_templates_v1.json', 'knowledge/risks/risk_catalog.json']
    sources = {n: sha(ROOT/n) for n in source_names}
    rows = []
    for identity, partition in APPROVED.items():
        meta = selected[identity]
        assert meta["partition"] == approved_rows[identity]["partition"] == partition
        assert meta["original_split"] in {"train", "validation"}
        path = (ROOT / 'code/external/AgentLeak' / meta["path"]).resolve()
        assert path.is_relative_to((ROOT / 'code/external/AgentLeak').resolve())
        assert sha(path) == meta["sha256"]
        row = {"id": identity, "partition": partition, "source_sha256": meta["sha256"]}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            checked = audit_trace(raw, meta, templates[meta["template_id"]])
            assert checked["candidate_complete_mas"] and not checked["issues"]
            calls, initial = raw["llm_calls"], raw["input"]
            prefix_keys = ("call_id", "system_prompt", "user_prompt", "response")
            coordinator = {k: calls[1][k] for k in prefix_keys}
            worker = {k: calls[2][k] for k in prefix_keys}
            # Both prefixes are constructed before the separately held future labels.
            early, _ = compile_prefix(initial, coordinator=coordinator)
            late, _ = compile_prefix(initial, coordinator=coordinator, worker=worker)
            labels = later_labels(initial, worker_text=calls[2]["response"], worker_record_id=calls[2]["call_id"],
                                  final_text=calls[3]["response"], final_record_id=calls[3]["call_id"])
            objects = protected_groups(initial)
            per_value = []
            for j, group in enumerate(objects):
                checks = {}
                for node, mi in (("work", 2), ("publish", 4)):
                    stored = raw["channel_messages"][mi]["leaked_fields"]
                    checks[node] = all(bool(labels[f"value-{j}:{node}"]["value"]) == (field in stored)
                                       for field in group["field_names"])
                assert all(checks.values())
                per_value.append(checks)
            summaries = []
            for phase, dag in (("early", early), ("late", late)):
                before = dag.sha256
                future = {n: v for n, v in labels.items() if n not in dag.observations}
                assert set(validated_labels(dag, future)) == set(dag.ids)-set(dag.observations)
                assert dag.sha256 == before
                available = {"configured-vault", calls[1]["call_id"]}
                if phase == "late": available.add(calls[2]["call_id"])
                assert {v["record_id"] for v in dag.observations.values()} == available
                assert all(v["sequence"] <= dag.prefix_sequence for v in dag.observations.values())
                assert not any(n.endswith(":publish") for n in dag.observations)
                if phase == "early": assert not any(n.endswith(":work") for n in dag.observations)
                summaries.append({"phase": phase, "dag_sha256": dag.sha256,
                    "workflow_nodes": len(dag.workflow["nodes"]), "workflow_edges": len(dag.workflow["edges"]),
                    "events": len(dag.ids), "observations": len(dag.observations),
                    "future_labels": len(future), "prefix_sequence": dag.prefix_sequence,
                    "observation_stage_counts": dict(Counter(dag.lookup[n]["node"] for n in dag.observations)),
                    "prefix_future_labels_separate": True, "prefix_unchanged_by_label_validation": True})
            row.update(status="passed", raw_response_bindings_verified=True, protected_groups=len(objects),
                all_stored_per_value_labels_consistent=True, stored_label_consistency_by_value=per_value,
                prefixes=summaries)
        except Exception as error:
            # No exception string: third-party diagnostics might include source text.
            row.update(status="failed", failure_type=type(error).__name__,
                       failure_reason="Existing validation/compilation contract rejected candidate; original data retained unchanged.")
        assert sha(path) == meta["sha256"]
        rows.append(row)
    assert sources == {n: sha(ROOT/n) for n in source_names}
    assert sha(report_path) == REPORT_SHA and sha(selection_path) == report["selection_sha256"]
    result = {"schema": "legacy-agentleak-approved-candidate-projection-check-v1", "records": rows,
        "candidate_count": len(rows), "passed": sum(r["status"] == "passed" for r in rows),
        "compiled_prefixes": sum(len(r.get("prefixes", [])) for r in rows),
        "new_api_calls": 0, "model_fits": 0, "risk_predictions": 0, "performance_scores": 0,
        "split_changed": False, "partial_candidates_opened": 0, "test_raw_files_opened": 0,
        "source_sha256": sources, "original_report_sha256": REPORT_SHA,
        "selection_sha256": report["selection_sha256"], "historical_artifacts_unchanged": True,
        "limitations": ["Compatibility check only; no new performance or deployment claim.",
            "Only four already-approved complete legacy MAS candidates; fit and validation remain separate, calibration remains unchanged.",
            "Provider finish_reason and provider response IDs are absent; local call binding does not certify provider completion or independence.",
            "Future labels are temporary checks against pre-existing successful-response labels; no labels are supplemented or persisted.",
            "Functional-stage sequences are not measured wall-clock warning times. These single-chain records do not supply repeated downstream continuations."]}
    write_new(output, result)
    print(json.dumps({"output": str(output), "sha256": sha(output), "passed": result["passed"],
                      "compiled_prefixes": result["compiled_prefixes"],
                      "records": [{"id": r["id"], "partition": r["partition"], "status": r["status"],
                                   "prefixes": r.get("prefixes", [])} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
