"""Same-prefix numerical inputs and metrics for the fixed two-field pilot.

No template/condition/task IDs, timestamps, final responses or labels enter
features. The full matrix is available to classifiers, not just its pooled OR.
Current authority is applied by the same exact shared-world gate for all rules
and classifiers. This pilot representation is not a variable-size DAG encoder.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from .branch_privacy_input import BranchPrivacyInput
from .privacy_model_independent import compile_packets
from .scoped_privacy_monitor import gated_distribution


def sample_from_prefix(prefix):
    return BranchPrivacyInput(prefix["observation"], available_journal_sequence=prefix["seq"], prefix_journal_sequence=prefix["seq"])


def field_features(sample):
    sample.validate()
    packet = sample.release.packet
    if len(packet["groups"]) != 2 or len(sample.workers) != 2:
        raise ValueError("this pilot baseline encodes exactly two fields and two workers")
    data = compile_packets([packet])
    global_features = np.r_[data["dc"].ravel(), sample.worker_presence.astype(float).ravel(),
        sample.declared_owner_mask.astype(float).ravel(), sample.release.visible_features.ravel()]
    # Full context + every branch/field. Target-row local values let a linear
    # baseline use the same conditional appearance cues without interactions.
    return np.c_[np.tile(global_features, (2, 1)), np.eye(2), data["dc"][0],
        sample.worker_presence, sample.declared_owner_mask, sample.release.visible_features]


def rule_distribution(sample, method):
    q = sample.worker_presence.any(axis=1).astype(float)
    if method == "visible_compliance_shared_gate":
        q *= sample.release.visible_features[:, 0]
    elif method != "persistence_shared_gate":
        raise ValueError("declared rule required")
    return gated_distribution(sample.release, q, .5)


def classifier_distribution(sample, estimator):
    q = estimator.predict_proba(field_features(sample))[:, 1]
    return gated_distribution(sample.release, q, .5)


def metrics(distributions, outcomes):
    dist = np.asarray(distributions, float)
    if not np.isfinite(dist).all() or np.any(dist < -1e-14) or not np.allclose(dist.sum(axis=1), 1., atol=1e-12):
        raise ValueError("proper violation count distributions required")
    result = {}
    for k in (1, 2):
        p = dist[:, k:].sum(axis=1)
        y = np.array([row["targets"][str(k)] for row in outcomes])
        clipped = np.clip(p, 1e-12, 1 - 1e-12)
        result[str(k)] = {"n": len(y), "positives": int(y.sum()), "brier": float(np.mean((p - y) ** 2)),
            "nll_clipped_1e_12": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
            "f1_at_half": float(f1_score(y, p >= .5, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, p)) if len(set(y)) == 2 else None}
    return result
