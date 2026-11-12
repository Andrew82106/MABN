import json

import pytest

from paperalpha_runtime.mas_annotation import (
    agreement_report,
    build_annotation_rows,
    cohen_kappa,
    evidence_set_agreement,
)


def episode(label=1):
    return {
        "episode_id": "family-secret-001",
        "agents": ["planner", "worker"],
        "events": [
            {"actor": "planner", "event_type": "input_received", "data_class": "untrusted", "sequence": 0},
            {"actor": "planner", "target": "worker", "event_type": "message_sent", "edge_confidence": 0.8, "sequence": 1},
            {"actor": "worker", "event_type": "tool_call", "tool": "shell.exec", "sequence": 2},
        ],
        "workflow_dag": {"nodes": ["planner", "worker"], "edges": [["planner", "worker"]]},
        "label": label,
        "hidden_fields": {"mechanism": "secret", "authorization_granted": False},
        "scenario_family": "should-not-leak",
    }


def test_packet_is_blind_and_keeps_opaque_evidence_ids():
    rows, source_index = build_annotation_rows(
        [episode()],
        [{"episode_id": "family-secret-001", "hierarchical_fusion": 0.99, "runtime_bn": 0.88}],
    )
    assert len(rows) == 1 and rows[0]["annotation_id"].startswith("mas-")
    assert rows[0]["annotation"] == {
        "system_violation": None,
        "violation_class": None,
        "first_violation_event_id": None,
        "evidence_event_ids": [],
        "rationale": "",
    }
    serialised = json.dumps(rows, sort_keys=True)
    for secret in ("should-not-leak", "authorization_granted", '"label"', "hierarchical_fusion", "runtime_bn", "family-secret-001"):
        assert secret not in serialised
    events = rows[0]["trace"]["events"]
    assert [event["evidence_id"] for event in events] == [
        f'{rows[0]["annotation_id"]}:event:0000',
        f'{rows[0]["annotation_id"]}:event:0001',
        f'{rows[0]["annotation_id"]}:event:0002',
    ]
    assert "edge_confidence" in events[1]
    assert source_index[0]["annotation_id"] == rows[0]["annotation_id"]
    assert "family-secret-001" not in json.dumps(source_index)


def test_pair_is_deterministic_and_projection_is_optional():
    first, _ = build_annotation_rows([episode()], salt="s", shuffle_seed=0)
    second, _ = build_annotation_rows([episode()], salt="s", shuffle_seed=0)
    assert first == second
    rows, _ = build_annotation_rows([episode()], projections=[])
    assert rows[0]["monitor_projection"] == {}


def test_cohen_kappa_and_evidence_agreement():
    result = cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0])
    assert result["n"] == 4
    assert result["observed_agreement"] == pytest.approx(0.75)
    assert result["cohens_kappa"] == pytest.approx(0.5)
    evidence = evidence_set_agreement([["a", "b"], ["c"]], [["a"], ["d"]])
    assert evidence["exact_set_agreement"] == pytest.approx(0.0)
    assert evidence["mean_jaccard"] == pytest.approx(0.25)


def test_agreement_requires_same_ids_and_reports_missing_values():
    a, _ = build_annotation_rows([episode()], salt="s")
    b, _ = build_annotation_rows([episode()], salt="s")
    evidence_id = a[0]["trace"]["events"][2]["evidence_id"]
    a[0]["annotation"].update(system_violation=True, violation_class="other", first_violation_event_id=evidence_id, evidence_event_ids=[evidence_id], rationale="tool event")
    b[0]["annotation"].update(system_violation=True, violation_class="other", first_violation_event_id=evidence_id, evidence_event_ids=[evidence_id], rationale="tool event")
    report = agreement_report(a, b)
    assert report["system_violation"]["cohens_kappa"] == pytest.approx(1.0)
    assert report["evidence_event_ids"]["mean_jaccard"] == pytest.approx(1.0)
    b[0]["annotation"]["system_violation"] = None
    report = agreement_report(a, b)
    assert report["system_violation"]["n"] == 0
    b[0]["annotation"]["system_violation"] = False
    with pytest.raises(ValueError, match="annotation ids differ"):
        agreement_report(a, [{**b[0], "annotation_id": "other"}])
