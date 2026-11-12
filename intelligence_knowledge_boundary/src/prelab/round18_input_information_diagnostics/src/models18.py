"""R18 fixed CPU classifiers; all fitted state uses fit_ix, never calibration.

The MLP is a fixed three-seed ensemble, with the same protocol seeds in every
fold. Its coupled Adam penalty includes biases. Hidden PCA is centered using
fit-only, label-free base weights, without whitening or feature standardization
before PCA; its 32 coordinates are appended to base before a joint scaler.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "models18_reuses_r13", ROOT.parent / "round13_generalization_diagnostics/src/run13.py")
r13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r13)

METHODS = ("mean_lr", "slots_lr", "shuffled_slots_lr", "mean_mlp",
           "mean_surface", "mean_alignment", "mean_both", "mean_hidden32")
WIDTHS = {"base": 785, "slots": 3144, "shuffled_slots": 3144,
          "surface": 8, "alignment": 16, "hidden": 3584}


def _array(designs, key, indices=None):
    value = np.asarray(designs[key])
    if value.ndim != 2 or value.shape[1] != WIDTHS[key]:
        raise ValueError(f"{key}: expected N x {WIDTHS[key]}, got {value.shape}")
    # Inspect values only after selection, so fit never consults held-out data.
    value = value if indices is None else value[indices]
    if not np.isfinite(value).all():
        raise ValueError(f"{key}: nonfinite selected features")
    return value


def _hidden_projection(hidden, base_weights, config, fold):
    pc = config["hidden_projection"]
    dim, iterations = int(pc["components"]), int(pc["n_iter"])
    if dim != 32 or iterations != 5 or min(hidden.shape) < dim:
        raise ValueError("Hidden projection requires 32 components and >=32 fit rows")
    normalized = np.asarray(base_weights, np.float64) / np.sum(base_weights)
    x = np.asarray(hidden, np.float64)
    center = normalized @ x
    centered = x - center
    weighted = centered * np.sqrt(normalized[:, None])
    total = float(np.einsum("ij,ij->", weighted, weighted))
    if not total > 0:
        raise ValueError("Hidden fit features have no weighted variance")
    seed = int(pc["seed"]) + int(fold)
    _, singular, components = randomized_svd(
        weighted, n_components=dim, n_iter=iterations, random_state=seed,
        flip_sign=True)
    variance = singular ** 2
    return {"mean": center, "components": components,
            "singular_values": singular, "total_variance": total,
            "explained_variance": variance,
            "explained_variance_ratio": variance / total,
            "explained_variance_ratio_sum": float(variance.sum() / total),
            "n_components": dim, "n_iter": iterations, "seed": seed,
            "whiten": False, "normalization": "base_weights / base_weights.sum()",
            "centering": "fit-only base-weighted hidden mean, no pre-PCA scaler",
            "fit_rows": len(hidden)}


def _raw(method, designs, indices=None, projection=None):
    if method == "slots_lr":
        return _array(designs, "slots", indices)
    if method == "shuffled_slots_lr":
        return _array(designs, "shuffled_slots", indices)
    blocks = [_array(designs, "base", indices)]
    if method in ("mean_surface", "mean_both"):
        blocks.append(_array(designs, "surface", indices))
    if method in ("mean_alignment", "mean_both"):
        blocks.append(_array(designs, "alignment", indices))
    if method == "mean_hidden32":
        if projection is None:
            raise ValueError("Missing fitted hidden projection")
        hidden = np.asarray(_array(designs, "hidden", indices), np.float64)
        blocks.append((hidden - projection["mean"]) @ projection["components"].T)
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    if any(len(block) != len(blocks[0]) for block in blocks):
        raise ValueError("Design arrays disagree on window count")
    return blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)


class SmallMLP(torch.nn.Module):
    def __init__(self, input_width, hidden=32, dropout=0.1):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(int(input_width), int(hidden)), torch.nn.ReLU(),
            torch.nn.Dropout(float(dropout)), torch.nn.Linear(int(hidden), 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def _probabilities(net, tx):
    net.eval()
    with torch.no_grad():
        if not len(tx):
            return np.empty(0, np.float64)
        return torch.cat([net(chunk).sigmoid() for chunk in tx.split(1024)]).numpy().astype(np.float64)


def _mlp_fit(z, y, weights, config):
    mc = dict(config["mlp"])
    if (int(mc["hidden"]), float(mc["dropout"]), int(mc["epochs"]),
        int(mc["batch_size"]), float(mc["learning_rate"])) != (32, 0.1, 80, 256, 0.003):
        raise ValueError("MLP settings disagree with fixed R18 design")
    seeds = [int(seed) for seed in mc["seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Exactly three distinct predeclared MLP seeds required")
    mass, count = float(config["target_loss_mass"]), len(y)
    decay = 1.0 / (float(config["C"]) * mass)
    tx = torch.as_tensor(np.ascontiguousarray(z), dtype=torch.float32, device="cpu")
    ty = torch.as_tensor(y, dtype=torch.float32, device="cpu")
    tw = torch.as_tensor(weights, dtype=torch.float32, device="cpu")
    members = []
    for seed in seeds:
        started = time.perf_counter()
        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        net = SmallMLP(z.shape[1], mc["hidden"], mc["dropout"]).cpu()
        # Adam's default decay is coupled L2, unlike AdamW. All parameters,
        # including both Linear biases, are in this one regularized group.
        optimizer = torch.optim.Adam(net.parameters(), lr=mc["learning_rate"],
                                     weight_decay=decay)
        history = []
        for epoch in range(1, int(mc["epochs"]) + 1):
            net.train()
            for ix in torch.randperm(count, generator=generator).split(mc["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    net(tx[ix]), ty[ix], reduction="none")
                loss = (tw[ix] * bce).mean() * count / mass
                loss.backward()
                optimizer.step()
            # Fixed full-fit diagnostics, with dropout disabled. No early stop,
            # held-out scores, or best-epoch selection participates in training.
            net.eval()
            with torch.no_grad():
                weighted_bce = 0.0
                for begin in range(0, count, 1024):
                    end = min(begin + 1024, count)
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        net(tx[begin:end]), ty[begin:end], reduction="none")
                    weighted_bce += float((tw[begin:end] * bce).double().sum()) / mass
                norm2 = float(sum(p.double().square().sum() for p in net.parameters()))
                regularization = 0.5 * decay * norm2
            objective = weighted_bce + regularization
            if not np.isfinite(objective):
                raise FloatingPointError(f"MLP nonfinite objective: seed={seed}, epoch={epoch}")
            history.append({"epoch": epoch, "weighted_bce": weighted_bce,
                            "l2_penalty": regularization, "objective": objective,
                            "parameter_norm_squared_including_bias": norm2})
        members.append({"seed": seed, "state_dict": {
            k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            "history": history, "final_epoch": int(mc["epochs"]),
            "final_fit_scores": _probabilities(net, tx),
            "seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in net.parameters())})
    return {"mlp_config": mc, "members": members, "weight_decay": decay,
            "optimizer": "Adam coupled L2 on all parameters including biases",
            "seed_policy": "same three protocol seeds in every fold; no fold offset",
            "minibatch_objective": "mean(loss_weights * BCE) * Nfit / target_loss_mass",
            "diagnostic_objective": "sum(loss_weights * BCE) / target_loss_mass + 0.5 * decay * sum(theta^2)",
            "training_dropout_in_diagnostic_curve": False,
            "early_stopping": False, "epoch_selection": "fixed final epoch",
            "aggregation": "arithmetic mean of all three probabilities",
            "final_fit_scores": np.mean([m["final_fit_scores"] for m in members], axis=0)}


def fit_model(method, designs, fit_ix, fit_rows, config, fold):
    """Fit one predeclared model; fit_rows are aligned one-for-one with fit_ix."""
    started = time.perf_counter()
    if method not in METHODS or method not in config["methods"]:
        raise ValueError(f"Method not predeclared: {method}")
    if float(config["C"]) != 0.01 or float(config["target_loss_mass"]) != 3854:
        raise ValueError("R18 C and loss mass must remain fixed")
    if int(config["classifier_seed"]) != 20260913 or int(config["lr_max_iter"]) != 2000:
        raise ValueError("Unexpected LR seed or iteration budget")
    fit_ix = np.asarray(fit_ix, dtype=np.int64)
    if fit_ix.ndim != 1 or len(fit_ix) != len(fit_rows) or not len(fit_ix):
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
        raise ValueError("Ineligible windows cannot be classifier training labels")
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        base, weight, factors = r13.weights(fit_rows, float(config["target_loss_mass"]))
        projection = None
        if method == "mean_hidden32":
            projection = _hidden_projection(_array(designs, "hidden", fit_ix), base, config, fold)
        raw = _raw(method, designs, fit_ix, projection)
        scaler = StandardScaler().fit(raw, sample_weight=base)
        z = scaler.transform(raw).astype(np.float32)
        model = {"method": method, "kind": "mlp" if method == "mean_mlp" else "lr",
                 "fold": int(fold), "scaler": scaler, "projection": projection,
                 "input_width": raw.shape[1], "C": float(config["C"]),
                 "target_loss_mass": float(config["target_loss_mass"]),
                 "base_weights": base, "loss_weights": weight, "class_factors": factors,
                 "base_weight_sum": float(base.sum()), "loss_weight_sum": float(weight.sum()),
                 "loss_mass_by_label": np.bincount(y, weights=weight, minlength=2),
                 "fit_ix": fit_ix.copy(), "fit_window_keys": keys, "fit_keys": keys,
                 "fit_groups": sorted({row["group_id"] for row in fit_rows}),
                 "fit_y": y.copy(), "scaler_weighting": "fit-only base weights, no class factors",
                 "cpu_threads": 4, "fit_only": True,
                 "capacity_note": "equal numeric C/loss mass does not equalize capacity across representations or LR/MLP"}
        if method == "mean_mlp":
            model.update(_mlp_fit(z, y, weight, config))
        else:
            classifier = LogisticRegression(
                C=config["C"], penalty="l2", solver="liblinear",
                max_iter=config["lr_max_iter"], random_state=config["classifier_seed"])
            classifier.fit(z, y, sample_weight=weight)
            if classifier.n_iter_.max() >= config["lr_max_iter"]:
                raise RuntimeError("LR hit its frozen iteration limit")
            model.update(model=classifier, classifier_seed=int(config["classifier_seed"]),
                         final_fit_scores=classifier.predict_proba(z)[:, 1].astype(np.float64))
        model["fit_seconds"] = time.perf_counter() - started
    return model


def _transformed(model, designs):
    raw = _raw(model["method"], designs, projection=model["projection"])
    if raw.shape[1] != model["input_width"]:
        raise ValueError("Prediction design width differs from fit")
    if not len(raw):
        return np.empty((0, model["input_width"]), dtype=np.float32)
    return model["scaler"].transform(raw).astype(np.float32)


def predict_members(model, designs):
    """Return 3 x N float64 fixed-seed probabilities, or None for an LR."""
    if model["kind"] != "mlp":
        return None
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        z = _transformed(model, designs)
        tx = torch.as_tensor(np.ascontiguousarray(z), dtype=torch.float32, device="cpu")
        mc = model["mlp_config"]
        scores = []
        for member in model["members"]:
            net = SmallMLP(model["input_width"], mc["hidden"], mc["dropout"]).cpu()
            net.load_state_dict(member["state_dict"])
            scores.append(_probabilities(net, tx))
        result = np.asarray(scores, dtype=np.float64)
        if result.shape != (3, len(z)) or not np.isfinite(result).all():
            raise ValueError("Invalid MLP member probabilities")
        return result


def predict(model, designs):
    """Predict every supplied window, preserving order and float64 dtype."""
    with threadpool_limits(limits=4):
        if model["kind"] == "mlp":
            scores = predict_members(model, designs).mean(axis=0)
        else:
            z = _transformed(model, designs)
            scores = model["model"].predict_proba(z)[:, 1] if len(z) else np.empty(0)
        result = np.asarray(scores, dtype=np.float64)
        if result.ndim != 1 or not np.isfinite(result).all():
            raise ValueError("Invalid model probabilities")
        return result
