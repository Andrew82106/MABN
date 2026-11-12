"""Constructed branch mechanisms only; never API samples or performance data."""
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.branched_publisher_data import cases, issuance, transitions, stage_request, prefix
from paperalpha_monitor.branch_privacy_input import BranchPrivacyInput


def fixture(*, current="revoked", visible="supported", shared=True, c=(0, 1),
            w=((1, 0), (0, 1)), owners=(0, 1), kinds=("canary", "number")):
    if current not in {"valid", "revoked", "root_unknown", "delegation_unknown", "opaque"}:
        raise ValueError("explicit current-authority mechanism required")
    if visible not in {"supported", "contradicted", "unresolved"}:
        raise ValueError("explicit visible-authority mechanism required")
    case = copy.deepcopy(cases()[0 if shared else 4])
    case["condition"] = "revoked_stale" if current == "revoked" else "valid_visible"
    for j, field in enumerate(case["tracked_fields"]):
        field["kind"] = kinds[j]
        case["field_owners"][field["field_names"][0]] = ("worker-a", "worker-b")[owners[j]]
    planner = "Constructed plan. " + " ".join(f["literal"] for j, f in enumerate(case["tracked_fields"]) if c[j])
    workers = {name: "Constructed workpaper. " + " ".join(f["literal"] for j, f in enumerate(case["tracked_fields"]) if w[j][i])
               for i, name in enumerate(("worker-a", "worker-b"))}
    raw = prefix(case, planner, workers, issuance(case) + transitions(case), stage_request(case, "publisher", workers=workers))
    # Visible claims and current evidence are deliberately varied separately.
    # Some combinations are epistemic mechanism checks, not service histories.
    if visible == "contradicted":
        raw["publisher_payload"]["cached_authority_records"] += transitions({**case, "condition": "revoked_refreshed"})
    elif visible == "unresolved":
        raw["publisher_payload"]["cached_authority_records"] = [r for r in issuance(case) if r["payload"]["parent_id"] is not None]
    scope = raw["pooled_scope"]
    if current in {"root_unknown", "delegation_unknown"}:
        scope["coverage_complete"] = False
        scope["complete_grant_ids"] = ["child-a", "child-b"] if current == "root_unknown" else ["root-a", "root-b"]
    elif current == "opaque":
        scope["authority_records"] = [r for r in scope["authority_records"] if r["payload"]["parent_id"] is not None]
    return BranchPrivacyInput(raw, available_journal_sequence=20, prefix_journal_sequence=20)
