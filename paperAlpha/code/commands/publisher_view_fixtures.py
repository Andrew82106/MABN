"""Constructed dual-view mechanism fixtures; never represented as API outputs."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.scoped_privacy_api import cases, issuance, authority_transition, observation, stage_request, STAGES
from paperalpha_monitor.scoped_privacy_monitor import compile_observation
from paperalpha_monitor.policy_gated_privacy import ScopedRelease
from paperalpha_monitor.publisher_view_bn import PublisherInput, ViewSeparatedRelease


def fixture(*, current="revoked", visible="supported", shared=True, stage=2, c=(1, 1), w=(1, 1)):
    if current not in {"valid", "revoked", "root_unknown", "delegation_unknown", "opaque"} or visible not in {"supported", "contradicted", "unresolved", "missing"}:
        raise ValueError("explicit mechanism fixture required")
    case = copy.deepcopy(cases()[0 if shared else 2])
    case["treatment"] = "stale_root" if current == "revoked" else "valid"
    ct = " ".join(f["literal"] for i, f in enumerate(case["tracked_fields"]) if c[i])
    wt = " ".join(f["literal"] for i, f in enumerate(case["tracked_fields"]) if w[i])
    obs = observation(case, ct, wt, issuance(case) + authority_transition(case), "full" if current in {"valid", "revoked", "opaque"} else "root_updates_missing")
    if current == "delegation_unknown":
        obs["complete_grant_ids"] = ["root-a", "root-b"]
    if current == "opaque":
        obs["authority_records"] = [r for r in obs["authority_records"] if r["payload"]["parent_id"] is not None]
    release = compile_observation(obs)
    if stage == 1:
        packet = copy.deepcopy(release.packet)
        packet["stage"] = 1
        for group in packet["groups"]:
            group["worker_present"] = None
        release = ScopedRelease(packet, release.graph, release.context, release.bindings)
    elif stage != 2:
        raise ValueError("explicit cutoff required")
    source = None
    if visible != "missing":
        payload = json.loads(stage_request(case, STAGES[2], wt)["input"][0]["content"])
        if visible == "contradicted":
            payload["cached_authority_records"] += authority_transition({**case, "treatment": "stale_root"})
        elif visible == "unresolved":
            payload["cached_authority_records"] = [r for r in payload["cached_authority_records"] if r["payload"]["parent_id"] is not None]
        source = PublisherInput.capture(payload, transport_observer="constructed-input", captured_journal_sequence=20, prefix_journal_sequence=20)
    return ViewSeparatedRelease(release, source, trusted_transport_observers=["constructed-input"])
