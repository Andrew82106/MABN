import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/commands'))
from paperalpha_monitor.compact_action_bn import CompactCapabilityBN
from paperalpha_monitor.capability_action_bn import PARAMETER_NAMES
from paperalpha_monitor.tool_capabilities import load_tool_registry
import diagnose_runtime_evidence_coverage as diagnostic


def events():
    return [
        {"sequence": 1, "event_type": "message_sent", "actor": "a", "target": "b", "content": "please handle request", "data_class": "unknown"},
        {"sequence": 2, "event_type": "tool_call", "actor": "b", "action_id": "call1", "tool": "authorize_payment", "arguments": {"amount": 10}},
        {"sequence": 3, "event_type": "tool_return_observed", "actor": "b", "action_id": "call1", "content": "not an authorization check"},
        {"sequence": 4, "event_type": "message_sent", "actor": "b", "target": "a", "content": "done", "data_class": "unknown"},
    ]


def parameters():
    return np.vstack([CompactCapabilityBN().theta, np.random.default_rng(6).normal(0., .6, (2, len(PARAMETER_NAMES)))])


def test_missing_source_mechanism_has_no_message_propagation_sensitivity():
    original = events()
    report, arrays = diagnostic.diagnose(original, parameters(), load_tool_registry())
    assert original == events()
    assert report["origin_variables"] == 0
    assert report["operation_counts"] == {"transmission": 2}
    assert report["maximum_information_state_probability"] == 0
    assert report["information_parameter_count"] == 37
    assert report["information_parameters_max_score_gradient"] == 0
    np.testing.assert_array_equal(arrays["scores"], arrays["no_message_scores"])
    np.testing.assert_array_equal(arrays["scores"], arrays["replaced_content_scores"])
    assert not report["labels_used"] and not report["parameter_fit_performed"]


def test_positive_control_source_enables_propagation_and_score_derivatives():
    source = {"sequence": 0, "event_type": "input_received", "actor": "a", "source_id": "test-only-source", "data_class": "untrusted"}
    report, arrays = diagnostic.diagnose([source, *events()], parameters(), load_tool_registry())
    assert report["origin_variables"] == 1
    assert report["maximum_information_state_probability"] > 0
    assert report["disable_message_propagation_max_score_difference"] > 1e-5
    assert report["information_parameters_max_score_gradient"] > 1e-5
    # No future event or variable allocation is used when evaluating a prefix.
    short = diagnostic.evaluate([source, *events()[:1]], parameters()[1], load_tool_registry())
    np.testing.assert_array_equal(short["scores"], arrays["scores"][1, :2])


def test_diagnostic_scores_gradients_match_finite_differences():
    theta, registry = parameters()[1], load_tool_registry()
    result = diagnostic.evaluate(events(), theta, registry)
    for name in ("source:unknown", "transmission:unknown:unknown:delegation=0",
                 "authorization:payment_effect:unknown:unknown", "head:bias"):
        index = list(PARAMETER_NAMES).index(name)
        plus, minus = theta.copy(), theta.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        numerical = (diagnostic.evaluate(events(), plus, registry)["scores"] -
                     diagnostic.evaluate(events(), minus, registry)["scores"]) / 2e-5
        np.testing.assert_allclose(result["score_gradients"][:, index], numerical, atol=1e-10, rtol=1e-5)


def test_observed_metadata_can_still_matter_without_information_propagation():
    original, changed = events(), events()
    changed[0]["data_class"] = "untrusted"
    theta, registry = parameters()[1], load_tool_registry()
    before = diagnostic.evaluate(original, theta, registry)
    after = diagnostic.evaluate(changed, theta, registry)
    assert np.max(np.abs(before["scores"] - after["scores"])) > 1e-5
    assert before["information_state_maxima"].max() == after["information_state_maxima"].max() == 0


@pytest.mark.parametrize("vectors", [[], np.zeros((1, 149)), np.full((1, 150), np.nan)])
def test_bad_parameter_probe_is_rejected(vectors):
    with pytest.raises(ValueError):
        diagnostic.diagnose(events(), vectors, load_tool_registry())


def test_content_replacement_preserves_non_text_payload_and_input():
    original = events()
    saved = copy.deepcopy(original)
    changed = diagnostic.replace_content(original)
    assert original == saved
    for before, after in zip(original, changed):
        assert {k: v for k, v in before.items() if k != "content"} == {k: v for k, v in after.items() if k != "content"}


def test_end_to_end_diagnostic_freezes_input_sources_and_numeric_arrays(tmp_path):
    import json
    import hashlib
    from paperalpha_runtime.journal import RuntimeJournal
    path = tmp_path / "journal.jsonl"
    journal = RuntimeJournal(path)
    journal.emit("run_started", {"agents": ["a", "b"], "external_endpoints": ["user"]})
    journal.emit("communication", {"actor": "a", "target": "b", "content": "example only"})
    journal.emit("run_finished", {"status": "completed"})
    output = tmp_path / "diagnostic"
    result = diagnostic.run(path, output)
    assert result["model_events"] == 1 and result["parameter_vectors"] == 9
    assert (output / "runtime_events.jsonl").read_bytes() == path.read_bytes()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["source_snapshot"].items():
        assert hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == metadata["sha256"]
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with np.load(output / "NUMERICS.npz", allow_pickle=False) as arrays:
        assert arrays["score_gradients"].shape == (9, 1, 150)
    with pytest.raises(FileExistsError):
        diagnostic.run(path, output)


def test_offline_diagnostic_does_not_accept_missing_run_end(tmp_path):
    from paperalpha_runtime.journal import RuntimeJournal
    path = tmp_path / "journal.jsonl"
    journal = RuntimeJournal(path)
    journal.emit("run_started", {"agents": ["a", "b"], "external_endpoints": []})
    journal.emit("communication", {"actor": "a", "target": "b", "content": "example only"})
    output = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="finalized capture"):
        diagnostic.run(path, output)
    assert not output.exists()
