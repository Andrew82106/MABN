"""Audit the first A2ASecBench version before treating it as model evidence.

This script does not train a model.  It checks source integrity, duplicate
episodes, label provenance, and produces deterministic family-held-out splits.
The existing Layer-2 score is reported as a transparent sanity baseline while
explicitly flagging that its features are derived from the same event stream
used to derive ``observed_violation``.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EPISODES = ROOT / 'data/processed/mas_adapter/a2asecbench/episodes.jsonl'
CANONICAL = ROOT / 'experiments/a2asecbench/runs/a2asecbench_v1'
OUT = ROOT / 'experiments/a2asecbench/runs/a2a_layer2_v1'
SPLIT_OUT = ROOT / 'data/processed/mas_adapter/a2asecbench/splits_v1.json'
FAMILIES = ["ASRF", "ATSI", "CO", "HOTF"]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(row: dict) -> float:
    f = row.get("features", {})
    return min(
        1.0,
        0.45 * min(float(row.get("event_count", 0)) / 20.0, 1.0)
        + 0.20 * bool(f.get("cross_agent_propagation"))
        + 0.20 * bool(f.get("permission_mismatch"))
        + 0.15 * bool(f.get("tool_side_effect")),
    )


def metrics(rows: list[dict], threshold: float = 0.5) -> dict:
    rows = [r for r in rows if r.get("observed_violation") is not None]
    pred = [score(r) >= threshold for r in rows]
    truth = [bool(r["observed_violation"]) for r in rows]
    tp = sum(p and y for p, y in zip(pred, truth))
    fp = sum(p and not y for p, y in zip(pred, truth))
    fn = sum(not p and y for p, y in zip(pred, truth))
    tn = sum(not p and not y for p, y in zip(pred, truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = read_jsonl(EPISODES)
    labeled = [r for r in rows if r.get("observed_violation") is not None]
    ids = [r.get("episode_id") for r in rows]
    duplicate_ids = len(ids) - len(set(ids))
    family_counts = Counter(str(r.get("attack_id")) for r in rows)
    label_counts = Counter(str(r.get("observed_violation")) for r in rows)

    # Family-held-out split is the honest first generalization protocol: no
    # attack family appears in both train and test for a split.
    splits = {}
    for held_out in FAMILIES:
        test = [r["episode_id"] for r in labeled if r.get("attack_id") == held_out]
        train = [r["episode_id"] for r in labeled if r.get("attack_id") != held_out]
        splits[f"leave_{held_out}_out"] = {"train": train, "test": test,
                                            "held_out_family": held_out}
    SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_OUT.write_text(json.dumps(splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    per_family = {fam: metrics([r for r in labeled if r.get("attack_id") == fam]) for fam in FAMILIES}
    overall = metrics(labeled)
    any_event = metrics([r for r in labeled if r.get("event_count", 0) > 0])

    # The current label is itself computed from positive aggregate counters or
    # event types.  This is useful for adapter sanity checking, but invalid as
    # an independent claim of detection performance.
    leakage = {
        "label_definition": "observed_violation(details.metrics/details.events)",
        "positive_event_types": ["artifact_executed", "task_store_pressure", "cycle_detected", "max_depth_exceeded"],
        "score_feature_event_types": ["artifact_executed", "task_store_pressure", "cycle_detected", "max_depth_exceeded", "uri_dereference"],
        "assessment": "potential_target_leakage_for_sanity_only",
    }
    canonical = {}
    for fam_dir in ("as", "asrf_eval", "atsi_eval", "co_eval", "hotf_eval"):
        path = CANONICAL / fam_dir / "run.jsonl"
        canonical[fam_dir] = {"exists": path.is_file(), "sha256": sha256(path) if path.is_file() else None,
                              "rows": len(read_jsonl(path)) if path.is_file() else 0}

    report = {
        "episodes": len(rows), "labeled_episodes": len(labeled),
        "null_label_episodes": len(rows) - len(labeled),
        "duplicate_episode_ids": duplicate_ids,
        "family_counts": dict(family_counts), "label_counts": dict(label_counts),
        "canonical_sources": canonical, "overall_sanity_score": overall,
        "per_family_sanity_score": per_family, "any_event_baseline": any_event,
        "leakage_audit": leakage, "split_file": str(SPLIT_OUT.relative_to(ROOT.parent)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# First-version data audit",
        "",
        "This is a data/protocol audit, not a claim that the current score is a trained detector.",
        "",
        f"- Episodes: **{len(rows)}**; labeled: **{len(labeled)}**; null labels excluded: **{len(rows)-len(labeled)}**",
        f"- Duplicate episode IDs: **{duplicate_ids}**",
        f"- Label counts: `{dict(label_counts)}`",
        f"- Current transparent Layer-2 sanity score: precision **{overall['precision']:.3f}**, recall **{overall['recall']:.3f}**, F1 **{overall['f1']:.3f}**",
        f"- Any-event baseline: precision **{any_event['precision']:.3f}**, recall **{any_event['recall']:.3f}**, F1 **{any_event['f1']:.3f}**",
        "",
        "## Validity warnings",
        "",
        "1. `observed_violation` is derived from the same runtime metrics/event types used by the score. Therefore the 0.727 F1 is an adapter sanity check with target leakage, not independent detector evidence.",
        "2. AS has 100 attack rows but no runtime events and null labels; it must not be mixed into supervised risk metrics.",
        "3. The canonical benchmark has attack/control families, but attack/control identity is not a security label. A real evaluation needs independently annotated benign and unsafe episodes or intervention outcomes.",
        "4. The generated split is leave-one-attack-family-out; it prevents the same attack family from appearing in train and test.",
        "",
        f"Split manifest: `{SPLIT_OUT.relative_to(ROOT.parent).as_posix()}`",
    ]
    (OUT / "AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
