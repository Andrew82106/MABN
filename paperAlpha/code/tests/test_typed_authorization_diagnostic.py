from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code/commands'))
from diagnose_typed_authorization_revision import diagnose
from paperalpha_monitor import TypedJointBN


def test_diagnostic_separates_v2_global_evidence_update_from_latent_revision():
    result = diagnose(TypedJointBN())
    assert result["same_authorization_state_under_opposite_later_checks"]
    assert not result["later_check_updates_latent"]
    clean, mismatch = result["cases"]
    assert clean["events"][0] == mismatch["events"][0]
    assert clean["global_permission_check_seen"] == mismatch["global_permission_check_seen"] == 1
    assert clean["global_permission_mismatch_seen"] == 0
    assert mismatch["global_permission_mismatch_seen"] == 1
    assert clean["new_latent_transitions"] == mismatch["new_latent_transitions"] == []
