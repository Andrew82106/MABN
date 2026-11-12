"""Handwritten paired inputs only; no SDK transport, private oracle or outcomes."""
import copy
import json
from uuid import uuid4

import pytest

from paperalpha_monitor.paired_forecast_observer import (
    PairedObserverError, public_prefix, project_paired_forecast_input,
)
from paperalpha_monitor.contract_forecast_projection import (
    FEATURE_NAMES, ForecastProjectionUncovered, project_forecast_input, to_forecast_input,
)
from paperalpha_runtime.journal import RuntimeEvidenceProjector
from test_contract_forecast_projection import Tape, FIN, EC, stage, claim, refresh_stage, refresh_claim
from test_harnessaudit_relational_evidence import update, trade, query, route


def configured_stage(tape, **kwargs):
    snapshot = stage(tape, **kwargs)
    prefix = snapshot["prefix_before"]
    prefix["public_task"]["agents"] = [
        {"role": binding["role"], "description": "declared duty " + binding["role"],
         "system_prompt": "PRIVATE CONFIGURED PROMPT " + binding["role"]}
        for binding in tape.bindings.values()]
    prefix.update(stage_purposes={}, output_delivery={"nodes": [snapshot["stage_id"]], "target": "user"},
                  tools=copy.deepcopy(snapshot["packet"]["tools"]))
    prefix["observability"]["state_visibility"] = []
    initial = copy.deepcopy(prefix)
    initial.update(sequence=0, journal=[], events=[], coverage=RuntimeEvidenceProjector().report())
    return refresh_stage(snapshot), initial


def paired(snapshot, request, tape, initial, contract=FIN, **kwargs):
    return project_paired_forecast_input(snapshot, request, tape.rows, contract, initial, **kwargs)


def test_private_prompt_intervention_does_not_change_public_inputs_before_behavior():
    tape = Tape("finance"); snapshot, initial = configured_stage(tape)
    request = claim(snapshot, tape)
    left = paired(snapshot, request, tape, initial)
    changed = copy.deepcopy(snapshot)
    changed["prefix_before"]["public_task"]["agents"][2]["system_prompt"] = "SECRET COMPROMISED ROLE PROMPT"
    changed["packet"]["instructions"] = "ACTUAL PRIVATE COMPROMISED INSTRUCTIONS"
    changed.update(condition="role_compromise", raw_path="private/cohort/attack", request_hash="secret")
    refresh_stage(changed)
    changed_claim = claim(changed, tape)
    assert changed_claim["request_sha256"] != request["request_sha256"]
    originals = copy.deepcopy((changed, changed_claim, tape.rows, initial))
    right = paired(changed, changed_claim, tape, initial)
    assert left == right
    assert (changed, changed_claim, tape.rows, initial) == originals
    encoded = json.dumps(right)
    for forbidden in ("SECRET COMPROMISED", "ACTUAL PRIVATE", "PRIVATE CONFIGURED", "request_sha256", "raw_path", "request_hash"):
        assert forbidden not in encoded
    assert all(set(a) == {"role", "description"} for a in right["visible_prefix"]["public_task"]["agents"])


def test_exact_raw_alignment_is_checked_before_any_sanitization():
    tape = Tape("finance"); snapshot, initial = configured_stage(tape); request = claim(snapshot, tape)
    request["request"]["system_instructions"] = "tampered private value"
    with pytest.raises(ForecastProjectionUncovered, match="claim_hash"):
        paired(snapshot, request, tape, initial)
    refresh_claim(request)
    with pytest.raises(ForecastProjectionUncovered, match="instructions_differ"):
        paired(snapshot, request, tape, initial)


def test_continuation_keeps_exact_request_horizon_features_and_proofs_not_future():
    tape = Tape("finance"); snapshot, initial = configured_stage(tape)
    identity = update(tape); request = claim(snapshot, tape, [identity], number=2)
    horizon = len(tape.rows)
    raw = project_forecast_input(snapshot, request, tape.rows, FIN)
    first = paired(snapshot, request, tape, initial)
    trade(tape)
    later = paired(snapshot, request, tape, initial)
    assert first == later
    assert first["features"] == raw["features"] and first["current_a"] == raw["current_a"] == 1
    assert first["prefix_horizon"] == horizon == first["visible_prefix"]["sequence"]
    assert first["evidence_path"] == raw["evidence_path"]
    assert first["evidence_path"]["request_feedback_links"][0]["terminal_sequence"] == horizon
    assert first["features"]["past_trigger_count"] == 0


@pytest.mark.parametrize("mutation,code", [("text", "feedback_text"), ("arguments", "request_binding"),
                                           ("missing", "missing_provider"), ("sequence", "invalid_visible_journal")])
def test_feedback_and_sequence_failures_are_not_hidden(mutation, code):
    tape = Tape("finance"); snapshot, initial = configured_stage(tape)
    identity = update(tape); request = claim(snapshot, tape, [identity])
    if mutation == "text":
        request["request"]["input"][-1]["output"] = "forged feedback"
    elif mutation == "arguments":
        request["request"]["input"][1]["arguments"] = '{"portfolio_id":"wrong"}'
    elif mutation == "missing":
        tape.rows.pop()
    else:
        tape.rows[2]["sequence"] = 99
    refresh_claim(request)
    with pytest.raises(ForecastProjectionUncovered, match=code):
        paired(snapshot, request, tape, initial)


def test_clocks_removed_but_sequence_and_actual_runtime_strings_preserved():
    tape = Tape("finance")
    tape.add("communication", {"actor": "risk", "target": "trade",
                              "content": "ACTUAL REPORT: system_prompt condition role_compromise"})
    tape.call("risk", "query_client_profile", {"system_prompt": "ACTUAL ARG", "condition": "ACTUAL ARG CONDITION"},
              "ACTUAL TOOL RESULT system_prompt role_compromise")
    for row in tape.rows:
        row.update(observed_at="private absolute time", monotonic_ns=987654321)
    snapshot, initial = configured_stage(tape)
    snapshot["prefix_before"]["events"] = [{"condition": "forged derived event"}]
    snapshot["prefix_before"]["coverage"] = {"condition": "forged coverage"}
    refresh_stage(snapshot)
    result = paired(snapshot, claim(snapshot, tape), tape, initial)
    view = result["visible_prefix"]
    assert [r["sequence"] for r in view["journal"]] == list(range(1, len(tape.rows) + 1))
    assert all(set(r) == {"schema_version", "sequence", "kind", "payload"} for r in view["journal"])
    assert [r["payload"] for r in view["journal"]] == [r["payload"] for r in tape.rows]
    text = json.dumps(view)
    assert "ACTUAL REPORT" in text and "ACTUAL ARG CONDITION" in text and "ACTUAL TOOL RESULT" in text
    assert "forged derived" not in text and "forged coverage" not in text
    assert "private absolute time" not in text and "987654321" not in text
    assert all("observed_at" in row for row in tape.rows)


def test_only_whitelisted_metadata_and_opaque_episode_association_are_exported():
    tape = Tape("finance"); snapshot, initial = configured_stage(tape); request = claim(snapshot, tape)
    snapshot["metadata"] = {"condition": "secret", "path": "secret", "unknown": "never pass through"}
    request["metadata"] = {"future_label": 1}
    identity = str(uuid4())
    result = paired(snapshot, request, tape, initial, episode_id=identity)
    assert result["metadata"] == {"contract_id": FIN["id"], "task_id": FIN["task_id"],
                                  "stage_id": "risk", "claim_number": 1, "episode_id": identity}
    sample = to_forecast_input(result)
    assert sample.features == result["features"] and sample.current_a == result["current_a"]
    assert sample.metadata == result["metadata"] and tuple(sample.features) == FEATURE_NAMES
    with pytest.raises(TypeError):
        paired(snapshot, request, tape, initial, metadata={"condition": "unapproved"})


@pytest.mark.parametrize("identity", ["control", "attack/task-1", "4ea26e58-aa62-11ef-a955-0242ac120002", 123, {}])
def test_nonopaque_or_time_based_episode_ids_rejected(identity):
    tape = Tape("finance"); snapshot, initial = configured_stage(tape)
    with pytest.raises(PairedObserverError, match="UUID4"):
        paired(snapshot, claim(snapshot, tape), tape, initial, episode_id=identity)


@pytest.mark.parametrize("field", ["goal", "role", "description", "workflow", "normative", "tools"])
def test_public_contract_and_legal_task_changes_are_not_silently_sanitized(field):
    tape = Tape("finance"); snapshot, initial = configured_stage(tape)
    raw = snapshot["prefix_before"]
    if field == "goal":
        raw["public_task"]["goal"] = "different assignment"
    elif field in {"role", "description"}:
        raw["public_task"]["agents"][0][field] = "changed"
    elif field == "workflow":
        raw[field]["edges"] = [["risk", "trade"]]
    elif field == "normative":
        raw[field]["rules"] = [{"condition": "not a legitimate declaration"}]
    else:
        raw[field] = []
    with pytest.raises(PairedObserverError, match="changed"):
        public_prefix(raw, initial)


@pytest.mark.parametrize("mutation", ["envelope", "payload", "gap", "template_events", "private_state"])
def test_unknown_journal_fields_and_invalid_prefixes_rejected(mutation):
    tape = Tape("finance"); snapshot, initial = configured_stage(tape); raw = snapshot["prefix_before"]
    if mutation == "envelope": raw["journal"][0]["condition"] = "private"
    elif mutation == "payload": raw["journal"][0]["payload"]["controlled_role"] = "private"
    elif mutation == "gap": raw["journal"][0]["sequence"] = 2
    elif mutation == "template_events": initial["sequence"] = 1; initial["journal"] = copy.deepcopy(raw["journal"])
    else: initial["observability"]["state_visibility"] = [{"table": "private"}]
    with pytest.raises(ValueError):
        public_prefix(raw, initial)


def test_empty_initial_observer_view_is_valid_and_condition_fields_are_not_copied():
    tape = Tape("finance"); _, initial = configured_stage(tape); raw = copy.deepcopy(initial)
    raw.update(condition="secret condition", path="private path", future_labels={"risk": 1})
    raw["public_task"]["agents"][0]["system_prompt"] = "private changed prompt"
    assert public_prefix(raw, initial) == public_prefix(initial, initial)


def test_cross_role_ecommerce_proof_survives_public_export():
    tape = Tape(); query(tape); route(tape)
    tape.add("communication", {"actor": "case", "target": "pay", "content": "real observed report"})
    reports = [{"sender": "case", "content": "real observed report", "sender_role": tape.bindings["case"]["role"],
                "sender_role_source": tape.bindings["case"]["source"]}]
    snapshot, initial = configured_stage(tape, actor="pay", edges=[["case", "pay"]], reports=reports)
    result = paired(snapshot, claim(snapshot, tape), tape, initial, EC)
    assert result["current_a"] == 1 and result["features"]["evidence_cross_role_count"] == 1
    assert result["features"]["predecessor_count"] == 1
    assert result["evidence_path"]["current_prerequisite"]["evidence_path"]
