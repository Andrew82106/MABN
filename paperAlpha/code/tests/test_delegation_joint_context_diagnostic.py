from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/commands'))
sys.path.insert(0, str(ROOT / 'code/src'))
from audit_delegation_joint_context import root_diagnostic, shared_root_fixture, replay
from paperalpha_monitor.delegation_conditioned_bn import DelegationConditionedBN
from test_qualified_delegation_cli import observed


def test_shared_root_pair_changes_only_one_parent_not_actions_or_local_evidence():
    left, a = shared_root_fixture(True)
    right, b = shared_root_fixture(False)
    assert a[:3] == b[:3] and a[4:] == b[4:]
    expected = {**a[3], "payload": {**a[3]["payload"], "parent_id": "root-b"}}
    assert b[3] == expected
    assert left.snapshots["action-worker-b"].evidence["candidates"][0]["path"] == ["root-a", "child-b"]
    assert right.snapshots["action-worker-b"].evidence["candidates"][0]["path"] == ["root-b", "child-b"]
    for identity in left.snapshots:
        assert left.snapshots[identity].features == right.snapshots[identity].features
    assert left.steps[-1].base.actions == right.steps[-1].base.actions


def test_current_independence_gap_is_not_mislabelled_as_measured_error():
    model = DelegationConditionedBN()
    model.fitted_targets = ["episode_outcome"]  # Protocol fixture only.
    result = root_diagnostic(model)
    assert all(r["maximum_difference"] == 0 for r in result["parameter_checks"])
    law = result["hypothetical_static_root_failure_law"]
    assert not law["empirically_estimated"]
    assert np.isclose(law["shared_root_any_failure"], .2)
    assert np.isclose(law["independent_roots_any_failure"], .36)


def test_channels_diagnostic_detects_absent_not_merely_zero_weight_text_input():
    config, records = observed("normal")
    model = DelegationConditionedBN()
    model.theta += np.random.default_rng(1904).normal(0, .3, len(model.theta))
    row = replay(model, config, records)[0]
    assert row["active_mechanisms"] == ["authorization"] and row["text_revisions"] == 0
    assert row["maximum_process_or_source_marginal"] == 0.
    assert row["maximum_information_parameter_derivative"] == 0.
    assert row["probability"] == replay(model, config, records, messages=False)[0]["probability"]
