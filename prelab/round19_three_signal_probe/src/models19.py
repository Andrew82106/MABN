"""Six fixed R19 LR designs, with the R18/R13 fit-only weighting protocol.

No PCA, neural network, feature selection, threshold search, or held-out fitting.
The caller supplies fit_rows aligned one-for-one with fit_ix. Predict preserves
the order and count of the supplied window designs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "models19_reuses_r13", ROOT.parent / "round13_generalization_diagnostics/src/run13.py")
r13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r13)

METHODS = ("base", "base_mmd", "base_ecs", "base_pks", "base_all", "signals_only")
WIDTHS = {"base": 785, "mmd": 2, "ecs": 4, "pks": 4}
BLOCKS = {"base": ("base",), "base_mmd": ("base", "mmd"),
          "base_ecs": ("base", "ecs"), "base_pks": ("base", "pks"),
          "base_all": ("base", "mmd", "ecs", "pks"),
          "signals_only": ("mmd", "ecs", "pks")}


def _raw(method, designs, indices=None):
    if method not in BLOCKS:
        raise ValueError(f"Unknown R19 method: {method}")
    blocks = []
    for name in BLOCKS[method]:
        value = np.asarray(designs[name])
        if value.ndim != 2 or value.shape[1] != WIDTHS[name]:
            raise ValueError(f"{name}: expected N x {WIDTHS[name]}, got {value.shape}")
        # Select first: even finiteness checks during fit use only fit rows.
        value = value if indices is None else value[indices]
        if not np.isfinite(value).all():
            raise ValueError(f"{name}: nonfinite selected features")
        if blocks and len(value) != len(blocks[0]):
            raise ValueError("Design blocks disagree on window count")
        blocks.append(value)
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)


def fit_model(method, designs, fit_ix, fit_rows, config, fold):
    """Fit exactly one predeclared LR on selected, resolved training windows."""
    started = time.perf_counter()
    if method not in METHODS or method not in config["methods"]:
        raise ValueError(f"Method not predeclared: {method}")
    if float(config["C"]) != 0.01 or float(config["target_loss_mass"]) != 3854:
        raise ValueError("R19 C and total loss mass must remain fixed")
    if int(config["classifier_seed"]) != 20260913 or int(config["lr_max_iter"]) != 2000:
        raise ValueError("Unexpected LR seed or iteration budget")
    fit_ix = np.asarray(fit_ix, dtype=np.int64)
    if fit_ix.ndim != 1 or not len(fit_ix) or len(fit_ix) != len(fit_rows):
        raise ValueError("Nonempty aligned fit_ix/fit_rows required")
    if len(np.unique(fit_ix)) != len(fit_ix) or fit_ix.min() < 0:
        raise ValueError("Fit indices must be unique and nonnegative")
    y = np.asarray([row["gold"] for row in fit_rows], dtype=np.int64)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Both resolved training classes required")
    keys = [row["window_key"] for row in fit_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Fit window keys must be unique")
    if any(not row.get("main_eligible", True) for row in fit_rows):
        raise ValueError("Ineligible windows cannot be used for fitting")
    with threadpool_limits(limits=4):
        base, weight, factors = r13.weights(fit_rows, float(config["target_loss_mass"]))
        raw = _raw(method, designs, fit_ix)
        scaler = StandardScaler().fit(raw, sample_weight=base)
        z = scaler.transform(raw).astype(np.float32)
        classifier = LogisticRegression(
            C=config["C"], penalty="l2", solver="liblinear",
            max_iter=config["lr_max_iter"], random_state=config["classifier_seed"])
        classifier.fit(z, y, sample_weight=weight)
        if classifier.n_iter_.max() >= config["lr_max_iter"]:
            raise RuntimeError("LR hit its fixed iteration limit")
        scores = classifier.predict_proba(z)[:, 1].astype(np.float64)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite fitted probabilities")
        return {"method": method, "kind": "lr", "fold": int(fold),
                "model": classifier, "scaler": scaler,
                "input_width": raw.shape[1], "design_blocks": list(BLOCKS[method]),
                "C": float(config["C"]), "classifier_seed": int(config["classifier_seed"]),
                "target_loss_mass": float(config["target_loss_mass"]),
                "base_weights": base, "loss_weights": weight, "class_factors": factors,
                "base_weight_sum": float(base.sum()), "loss_weight_sum": float(weight.sum()),
                "loss_mass_by_label": np.bincount(y, weights=weight, minlength=2),
                "fit_ix": fit_ix.copy(), "fit_window_keys": keys, "fit_keys": keys,
                "fit_y": y.copy(), "fit_groups": sorted({row["group_id"] for row in fit_rows}),
                "coefficients_standardized": classifier.coef_.copy(),
                "intercept_standardized": classifier.intercept_.copy(),
                "iterations": classifier.n_iter_.copy(), "lr_max_iter": int(config["lr_max_iter"]),
                "final_fit_scores": scores, "cpu_threads": 4, "fit_only": True,
                "scaler_weighting": "fit-only base weights; no class factors",
                "capacity_note": "equal C/loss mass does not equalize the capacity of different feature sets",
                "fit_seconds": time.perf_counter() - started}


def predict(model, designs):
    """Return finite float64 probabilities for every supplied window."""
    with threadpool_limits(limits=4):
        raw = _raw(model["method"], designs)
        if raw.shape[1] != model["input_width"]:
            raise ValueError("Prediction design width differs from fit")
        if not len(raw):
            return np.empty(0, dtype=np.float64)
        z = model["scaler"].transform(raw).astype(np.float32)
        result = np.asarray(model["model"].predict_proba(z)[:, 1], dtype=np.float64)
        if result.shape != (len(raw),) or not np.isfinite(result).all():
            raise ValueError("Invalid LR probabilities")
        return result
