from pathlib import Path
import importlib.util


SCRIPT = Path(__file__).resolve().parents[2] / 'code/commands/evaluate_groupcv_v1.py'
SPEC = importlib.util.spec_from_file_location("groupcv_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_public_event_terminal_fields_are_rejected():
    trace = {"episode_id": "ep-1", "events": [{"actor": "a0", "external_effect": True}]}
    try:
        MODULE.validate(trace)
    except ValueError as exc:
        assert "evaluation-only" in str(exc)
    else:
        raise AssertionError("event-level terminal labels must be rejected")


def test_permission_mask_is_local_and_removes_both_flags():
    trace = {"episode_id": "ep-1", "events": [{"permission_mismatch": True, "permission_denied": True, "actor": "a0"}]}
    masked = MODULE.mask_permission(trace)
    assert "permission_mismatch" not in masked["events"][0]
    assert "permission_denied" not in masked["events"][0]
    assert "permission_mismatch" in trace["events"][0]
