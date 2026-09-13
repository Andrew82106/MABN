"""Post-hoc, fit-only diagnostics for the frozen V2 pilot.

This script never changes the frozen feature bundle or the registered evaluator.
Its outputs are descriptive and must not be used to retune V2 on the same pilot.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVALUATOR = HERE / "evaluate_forced_evidence_quote_probe_v2.py"
OUT = ROOT / "research/forced_evidence_quote_probe_v2/FIT_ONLY_DIAGNOSTICS.json"
EXPECTED_EVALUATOR_SHA256 = (
    "a80e1ac7a107d96e32dd55e47e4c8535c5e7a189e491335f519acb604883ac8f"
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_evaluator():
    if sha256(EVALUATOR) != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError("V2 evaluator identity changed")
    spec = importlib.util.spec_from_file_location("forced_quote_v2_evaluator", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.assert_runtime_contract()
    module.assert_v1_scoring_parity()
    return module


def counts(y: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(y)),
        "positive": int(y.sum()),
        "positive_rate": float(y.mean()),
    }


def f1_recall(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    yb = np.asarray(y, dtype=bool)
    pb = np.asarray(prediction, dtype=bool)
    tp = int(np.sum(yb & pb))
    fp = int(np.sum(~yb & pb))
    fn = int(np.sum(yb & ~pb))
    f1 = 2.0 * tp / max(1, 2 * tp + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return float(f1), float(precision), float(recall)


def best_f1(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    candidates = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)]
    best = None
    for threshold in candidates:
        f1, precision, recall = f1_recall(y, scores >= threshold)
        key = (f1, precision, float(threshold))
        if best is None or key > best[0]:
            best = (key, threshold, f1, precision, recall)
    assert best is not None
    return {
        "descriptive_pooled_OOF_F1opt": float(best[2]),
        "precision": float(best[3]),
        "recall": float(best[4]),
        "threshold": float(best[1]),
    }


def outer_oof_claim_scores(module, condition) -> tuple[np.ndarray, list[dict]]:
    scores = np.full(len(condition.y), np.nan, dtype=np.float64)
    audits = []
    for fold in module.FOLDS:
        train = condition.folds != fold
        valid = condition.folds == fold
        fold_scores, audit = module.fit_predict(
            condition, train, valid, module.CONDITION_C[condition.name]
        )
        scores[valid] = fold_scores
        audits.append(audit)
    if not np.isfinite(scores).all():
        raise RuntimeError("incomplete claim OOF scores")
    return scores, audits


def oriented_univariate_ap(y: np.ndarray, matrix: np.ndarray) -> dict[str, float | int]:
    best = (-1.0, -1, 1)
    finite_columns = 0
    for column in range(matrix.shape[1]):
        values = np.asarray(matrix[:, column], dtype=np.float64)
        if not np.isfinite(values).all() or np.all(values == values[0]):
            continue
        finite_columns += 1
        for direction in (1, -1):
            ap = float(average_precision_score(y, direction * values))
            if ap > best[0]:
                best = (ap, column, direction)
    return {
        "nonconstant_columns": finite_columns,
        "best_full_fit_oriented_AP": best[0],
        "column": best[1],
        "direction": best[2],
    }


def run(output: Path) -> None:
    module = load_evaluator()
    receipt, clean_rows = module.load_frozen_features()
    geometry = module.load_real_gold_geometry(receipt, clean_rows)
    conditions = module.make_conditions(receipt, geometry)

    claim_oracle = module.project_to_windows(
        conditions["P3"], geometry.claim_y.astype(np.float64),
        np.ones(len(geometry.window_y), dtype=bool),
    )
    oracle_prediction = claim_oracle >= 0.5
    oracle_f1, oracle_precision, oracle_recall = f1_recall(
        geometry.window_y, oracle_prediction
    )
    answer_oracle = module.aggregate_answers(claim_oracle, geometry) >= 0.5
    answer_f1, answer_precision, answer_recall = f1_recall(
        geometry.answer_y, answer_oracle
    )

    condition_results = {}
    for name in ("P1", "P2", "P3"):
        condition = conditions[name]
        scores, audits = outer_oof_claim_scores(module, condition)
        fold_ap = {
            str(fold): float(average_precision_score(
                condition.y[condition.folds == fold], scores[condition.folds == fold]
            ))
            for fold in module.FOLDS
        }
        condition_results[name] = {
            "claim_outer_OOF_AP": float(average_precision_score(condition.y, scores)),
            "claim_outer_OOF_AUROC": float(roc_auc_score(condition.y, scores)),
            **best_f1(condition.y, scores),
            "per_fold_claim_AP": fold_ap,
            "univariate_full_fit_diagnostic": oriented_univariate_ap(
                condition.y, condition.X
            ),
            "outer_fit_count": len(audits),
        }

    p3 = receipt.arrays["P3"]
    exact = ((p3[:, 0] == 1.0) & (p3[:, 1] == 1.0) & (p3[:, 3] == 1.0))
    report = {
        "status": "fit_only_posthoc_diagnostic_complete",
        "warning": "Descriptive only; V2 is frozen and must not be retuned on this pilot.",
        "evaluator_sha256": sha256(EVALUATOR),
        "evaluation_sha256": sha256(
            ROOT / "results/forced_evidence_quote_probe_v2/EVALUATION.json"
        ),
        "feature_freeze_sha256": receipt.freeze_sha256,
        "labels": {
            "claims": counts(geometry.claim_y),
            "windows": counts(geometry.window_y),
            "answers": counts(geometry.answer_y),
        },
        "claim_geometry_oracle": {
            "window_F1": oracle_f1,
            "window_precision": oracle_precision,
            "window_recall": oracle_recall,
            "answer_F1": answer_f1,
            "answer_precision": answer_precision,
            "answer_recall": answer_recall,
        },
        "conditions": condition_results,
        "P3_exact_quote": {
            "count": int(exact.sum()),
            "rate": float(exact.mean()),
            "risk_rate_exact": float(geometry.claim_y[exact].mean()),
            "risk_rate_nonexact": float(geometry.claim_y[~exact].mean()),
        },
        "calibration_used_for_training_or_scoring": False,
        "official_test_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refuse overwrite: {output}")
    pending = output.with_suffix(output.suffix + ".pending")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output.resolve())
