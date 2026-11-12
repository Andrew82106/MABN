"""Recompute identity-free training objectives, selection and validation scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np


def verify(run, out):
    if out.exists():
        raise ValueError("new audit output required")
    sys.path[:0] = [str(run / "source_snapshot/code/src"), str(run / "source_snapshot/code/commands")]
    from evaluate_privacy_flow import sha, write
    from evaluate_privacy_supervision_controls import guard_old_test, score_details
    from verify_privacy_development import close, stage_reference
    from paperalpha_monitor.privacy_model_independent import (ModelIndependentPrivacyBN, compile_packets,
        direct_features, graph_features, KINDS)
    from paperalpha_monitor.privacy_supervision_controls import outcome_matrix, object_predictions
    plan = json.loads((run / "PLAN.json").read_text(encoding="utf-8"))
    attempts = guard_old_test(Path(plan["original_run"]).resolve())
    report = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
    for name, expected in report["artifacts_sha256"].items():
        if sha(run / name) != expected:
            raise ValueError("frozen artifact mismatch: " + name)
    train, val = [json.loads((run / f"{split}.json").read_text(encoding="utf-8")) for split in ("train", "validation")]
    assert not {r["template_id"] for r in train} & {r["template_id"] for r in val}
    assert all("model" not in r and "model" not in p for r in train + val for p in r["prefixes"].values())
    tp = [r["prefixes"]["2"] for r in train]
    data = compile_packets(tp)
    labels = outcome_matrix(data, [r["final_value_labels"] for r in train])
    candidates = json.loads((run / "CANDIDATES.json").read_text(encoding="utf-8"))
    selected = json.loads((run / "SELECTED.json").read_text(encoding="utf-8"))
    counts = {"fitted_bn": 0, "validation_candidates": 0}
    losses, gradients, scalar_errors, selections = [], [], [], {"bn": {}, "classifiers": {"1": {}, "2": {}}}
    for row in candidates:
        if "optimizer" in row:
            m = ModelIndependentPrivacyBN.restore(json.loads((run / row["file"]).read_text(encoding="utf-8")))
            loss, grad = m.loss_gradient(m.theta, data, labels)
            losses.append(abs(loss - row["optimizer"]["loss"]))
            assert losses[-1] < 1e-10 and m.optimizer["success"]
            direction = np.random.default_rng(76).normal(size=len(m.theta))
            direction /= np.linalg.norm(direction)
            eps = 1e-5
            numerical = (m.loss_gradient(m.theta + eps * direction, data, labels)[0] - m.loss_gradient(m.theta - eps * direction, data, labels)[0]) / (2 * eps)
            gradients.append(abs(numerical - grad @ direction))
            assert gradients[-1] < 1e-7
            counts["fitted_bn"] += 1
        elif row["method"].startswith("bn_"):
            pool = [r for r in candidates if r.get("method") == row["method"] and r.get("ridge") == row["ridge"] and "optimizer" in r and r["optimizer"]["success"]]
            assert min(pool, key=lambda r: (r["optimizer"]["loss"], r["file"]))["file"] == row["training_best_file"]
            m = ModelIndependentPrivacyBN.restore(json.loads((run / row["training_best_file"]).read_text(encoding="utf-8")))
            scores = {s: score_details(val, m.predict_details([r["prefixes"][s] for r in val])) for s in ("1", "2")}
            close(scores, row["validation"])
            for s in ("1", "2"):
                for packet in [r["prefixes"][s] for r in val[:8]]:
                    a = stage_reference(m, packet, lambda p: compile_packets([p])["context"][0], KINDS)
                    b = m.predict_details([packet])["probabilities"][0]
                    scalar_errors.append(abs(a - b))
            counts["validation_candidates"] += 1
        else:
            s = row["stage"]
            packets = [r["prefixes"][s] for r in val]
            m = joblib.load(run / row["file"])
            if row["method"] == "field_histgb":
                dv = direct_features(packets)
                q = np.zeros(dv["mask"].shape)
                q[dv["mask"]] = m.predict_proba(dv["design"][dv["mask"]])[:, 1]
                detail = {"probabilities": object_predictions(q, dv["mask"]),
                    "field_marginals": [q[i, dv["mask"][i]].tolist() for i in range(len(val))]}
            else:
                detail = {"probabilities": m.predict_proba(graph_features(packets))[:, 1]}
            close(score_details(val, detail), row["validation"])
            counts["validation_candidates"] += 1
    for method in selected["bn"]:
        choices = [r for r in candidates if r["method"] == method and "validation" in r]
        chosen = min(choices, key=lambda r: (np.mean([r["validation"][s]["endpoint"]["brier"] for s in ("1", "2")]), r["training_best_file"]))
        selections["bn"][method] = chosen["training_best_file"]
        for s in ("1", "2"):
            close(chosen["validation"][s], report["validation_scores"][s][method])
    for s in ("1", "2"):
        for method in selected["classifiers"][s]:
            choices = [r for r in candidates if r["method"] == method and r.get("stage") == s]
            chosen = min(choices, key=lambda r: (r["validation"]["endpoint"]["brier"], r["file"]))
            selections["classifiers"][s][method] = chosen["file"]
            close(chosen["validation"], report["validation_scores"][s][method])
    assert selections == selected
    assert max(scalar_errors) < 1e-12
    frozen = json.loads((run / "FROZEN.json").read_text(encoding="utf-8"))
    assert frozen["selected"] == selected and frozen["training_traces"] == len(train)
    assert abs(frozen["training_prevalence"] - np.mean([r["y"] for r in train])) < 1e-12
    write(out / "REPORT.json", {"run": str(run), "parent_report_sha256": sha(run / "REPORT.json"),
        "artifact_hashes_checked": len(report["artifacts_sha256"]), "counts": counts, "max_objective_error": max(losses),
        "max_directional_gradient_error": max(gradients), "max_scalar_probability_error": max(scalar_errors),
        "validation_scores_and_selection_recomputed": True, "old_test_open_attempts": attempts,
        "scope": "Numerical and selection audit, not independent generalization evidence"})
    print(json.dumps({"verified": str(out), "counts": counts, "max_scalar_error": max(scalar_errors)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve(), args.out.resolve())
