"""Frozen, post-diagnosis attribute experiment on existing API development data.

No new independent test or API collection. Every exposed validation result is
descriptive; no model selection. Standalone logit uses the exact same inputs,
labels, regularization and scoped policy truth table as the BN.
"""
from pathlib import Path
import argparse
from itertools import product, combinations
import json
import math
import re
import shutil
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_exclusive_purchase import read, write, sha, now
from evaluate_exclusive_purchase import reconstruct, policy_truth, scalar_choice_worlds
from paperalpha_monitor.choice_attribute_bn import AttributeChoiceWorkflowRiskBN, FEATURES, attribute_rows
from paperalpha_monitor.exclusive_dispatch_bn import ExclusiveDispatchWorkflowRiskBN
from paperalpha_runtime.purchase_attribute_adapter import compile_attribute_input
from scalar_binary_graph import evaluate

MASKS = {"full": list(range(10)), "cost_only": [0, 1, 2, 3, 6, 7],
         "warranty_only": [0, 1, 2, 4, 5, 8, 9]}


def reference_features(prefix, sample):
    """Independent catalogue extraction for this source's fixed English grammar.

    Intentionally rejects unsupported descriptions rather than calling the
    production parser. Used only on the original one/three-year catalogue.
    """
    priority = json.loads(prefix["actual_planner_text"])["priority"]
    economy, quality = int(priority == "economy"), int(priority == "quality")
    matrix = []
    for p in sample.proposals:
        branch = next(b for b in prefix["branches"] if b["actor"] == p["binding"]["actor"])
        item = next(v for v in branch["catalogue"] if v["sku"] == p["binding"]["arguments"]["sku"])
        tokens = re.findall(r"\b(one|three)-year warranty\b", item["description"])
        if len(tokens) != 1:
            raise ValueError("independent replay parser supports original literal one/three-year claims only")
        warranty = {"one": 1 / 3, "three": 1.}[tokens[0]]
        cost = float(item["price"]) / float(branch["single_limit"])
        matrix.append([1, economy, quality, cost, warranty, 0, economy * cost,
                       quality * cost, economy * warranty, quality * warranty])
    return np.asarray(matrix, float)


def reference_objective(theta, rows, penalty):
    loss = 0.; gradient = np.zeros(10)
    for row in rows:
        s, x = row["sample"], row["x"]
        for g in s.choice_groups:
            indices = [s.action_ids.index(a) for a in g["action_ids"]]
            design = np.vstack((np.zeros(10), x[indices]))
            logits = design @ theta; logp = logits - logsumexp(logits)
            selected = [i + 1 for i, a in enumerate(g["action_ids"]) if row["label"]["dispatch"][a]]
            if len(selected) > 1:
                raise ValueError("impossible actual choice")
            selected = selected[0] if selected else 0
            loss -= logp[selected]; gradient += np.exp(logp) @ design - design[selected]
    return float(loss / len(rows) + penalty * (theta @ theta)), gradient / len(rows) + 2 * penalty * theta


def reference_worlds(sample, x, theta):
    weights = []
    for g in sample.choice_groups:
        z = [0.] + [float(x[sample.action_ids.index(a)] @ theta) for a in g["action_ids"]]
        unnormalized = [math.exp(v - max(z)) for v in z]
        weights.append([v / sum(unnormalized) for v in unnormalized])
    worlds = list(product((0, 1), repeat=len(sample.action_ids))); probabilities = []
    for state in worlds:
        mass = 1.
        for g, p in zip(sample.choice_groups, weights):
            chosen = [i + 1 for i, a in enumerate(g["action_ids"]) if state[sample.action_ids.index(a)]]
            mass *= 0. if len(chosen) > 1 else p[chosen[0] if chosen else 0]
        probabilities.append(mass)
    return np.asarray(worlds), np.asarray(probabilities)


def support_diagnosis(rows):
    all_x, contrasts = [], []; abstentions = 0; slots = 0; alias_pairs = 0
    counts = {}
    for row in rows:
        s, x = row["sample"], row["x"]
        priority = json.loads(row["prefix"]["actual_planner_text"])["priority"]
        counts.setdefault(priority, {"slots": 0, "choices": {}})
        for g in s.choice_groups:
            ix = [s.action_ids.index(a) for a in g["action_ids"]]; slots += 1
            all_x.extend(x[ix])
            contrasts.extend(x[j] - x[ix[0]] for j in ix[1:])
            alias_pairs += sum(np.array_equal(x[i], x[j]) for i, j in combinations(ix, 2))
            chosen = [a for a in g["action_ids"] if row["label"]["dispatch"][a]]
            abstentions += int(not chosen); counts[priority]["slots"] += 1
            key = "abstention" if not chosen else str(row["sample"].proposals[s.action_ids.index(chosen[0])]["binding"]["arguments"]["sku"])
            counts[priority]["choices"][key] = counts[priority]["choices"].get(key, 0) + 1
    x, delta = np.asarray(all_x), np.asarray(contrasts)
    # Known from the *input catalogue*: a +30/90 cost change always accompanies
    # a +2/3 warranty change. This null direction changes both alleged tastes.
    witness = np.zeros(10); witness[3] = 1.; witness[4] = -.5
    return {"train_flows": len(rows), "slots": slots, "observed_abstentions": abstentions,
            "feature_columns": list(FEATURES), "full_choice_design_rank": int(np.linalg.matrix_rank(x)),
            "conditional_on_dispatch_contrast_rank": int(np.linalg.matrix_rank(delta)),
            "conditional_contrast_singular_values": np.linalg.svd(delta, compute_uv=False).tolist(),
            "still_identical_alternative_pairs": int(alias_pairs), "counts": counts,
            "conditional_cost_warranty_null_direction": witness.tolist(),
            "max_null_contrast": float(max(abs(delta @ witness))),
            "interpretation": "No abstention labels and price/warranty co-vary. Distinct alternatives fix aliasing but do not separately identify causal price/quality preferences; full-design rank alone is insufficient."}


def summarize(records, split):
    selected = [r for r in records if r["split"] == split]
    risks = np.asarray([r["risk_probability"] for r in selected]); y = np.asarray([r["risk_label"] for r in selected])
    slots = [s for r in selected for s in r["slots"]]
    return {"n": len(selected), "positive_flows": int(y.sum()), "risk_brier": float(np.mean((risks - y) ** 2)),
            "tp": int(((risks >= .5) & (y == 1)).sum()), "fn": int(((risks < .5) & (y == 1)).sum()),
            "fp": int(((risks >= .5) & (y == 0)).sum()), "tn": int(((risks < .5) & (y == 0)).sum()),
            "slot_count": len(slots), "slot_accuracy": float(np.mean([s["correct"] for s in slots])),
            "slot_brier": float(np.mean([s["brier"] for s in slots])),
            "slot_log_loss": float(np.mean([s["log_loss"] for s in slots])),
            "exact_flow_choices": sum(all(s["correct"] for s in r["slots"]) for r in selected)}


def prepare(source, destination):
    if not (source / "COLLECTION.json").exists():
        raise ValueError("existing immutable API source required")
    destination.mkdir(parents=True, exist_ok=False)
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'code/src').rglob("*.py")]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'knowledge/templates').glob("*.json")]
    names += ['knowledge/risks/risk_catalog.json', 'code/commands/collect_exclusive_purchase.py', 'code/commands/evaluate_exclusive_purchase.py',
              'code/commands/scalar_binary_graph.py', 'code/commands/evaluate_choice_attributes.py', 'code/tests/test_choice_attribute_bn.py',
              'code/tests/test_choice_attribute_experiment.py']
    for name in names:
        p = destination / "source" / name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, p)
    plan = {"schema": "attribute-choice-development-v1", "created_at_utc": now(), "source_run": str(source),
            "collection_sha256": sha(source / "COLLECTION.json"), "collection_plan_sha256": sha(source / "PLAN.json"),
            "source_sha256": {n: sha(destination / "source" / n) for n in sorted(names)},
            "regularization": [.001, .03], "feature_masks": MASKS, "max_iter": 500,
            "evaluation_status": "Post-diagnosis development replay; all 3 validation completions were previously exposed; no fresh test.",
            "fit_protocol": "Fit only original 12 training flows. All models fixed before scoring. Report both penalties and both attribute ablations; no winner selection.",
            "new_api_calls": 0, "new_default_selected": False, "old_test_read": False}
    write(destination / "PLAN.json", plan)
    print(json.dumps({"prepared": str(destination), "plan_sha256": sha(destination / "PLAN.json")}), flush=True)


def run(destination):
    plan = read(destination / "PLAN.json"); source = Path(plan["source_run"])
    if ROOT != (destination / "source").resolve():
        raise ValueError("execute frozen evaluator")
    for name, h in plan["source_sha256"].items():
        if sha(ROOT / name) != h: raise ValueError("frozen source changed: " + name)
    if sha(source / "COLLECTION.json") != plan["collection_sha256"] or sha(source / "PLAN.json") != plan["collection_plan_sha256"]:
        raise ValueError("raw API collection changed")
    if (destination / "FIT.json").exists() or (destination / "models").exists():
        raise ValueError("no overwriting fitted results")
    _, original, failed = reconstruct(source)
    rows = []; feature_error = 0.
    for r in original:
        pre = read(source / "runs" / r["case"]["id"] / "PREFIX.json")["value"]
        s = compile_attribute_input(pre); x = reference_features(pre, s)
        error = float(np.max(abs(x - attribute_rows(s)[0]))); feature_error = max(feature_error, error)
        if error > 1e-12: raise ValueError("attribute extraction differs from independent replay")
        rows.append({**r, "sample": s, "prefix": pre, "x": x})
    train = [r for r in rows if r["case"]["split"] == "train"]
    write(destination / "TRAINING_SUPPORT.json", support_diagnosis(train))
    models = {}; standalone = {}
    for penalty in plan["regularization"]:
        for name, cls in (("attribute_bn", AttributeChoiceWorkflowRiskBN), ("old_cue_bn", ExclusiveDispatchWorkflowRiskBN)):
            m = cls(regularization=penalty).fit([r["sample"] for r in train],
                    [{"dispatch": r["label"]["dispatch"]} for r in train], max_iter=plan["max_iter"])
            if not m.optimizer["success"]: raise ValueError("BN optimizer did not converge")
            key = f"{name}_{penalty}"; models[key] = m; write(destination / "models" / (key + ".json"), m.parameters())
        for mask_name, mask in plan["feature_masks"].items():
            result = minimize(lambda t: reference_objective(t, train, penalty), np.zeros(10), jac=True,
                              method="L-BFGS-B", bounds=[(-12., 12.) if i in mask else (0., 0.) for i in range(10)],
                              options={"maxiter": plan["max_iter"], "ftol": 1e-12, "gtol": 1e-7})
            if not result.success: raise ValueError("standalone attribute logit did not converge")
            key = f"standalone_{mask_name}_{penalty}"; standalone[key] = result.x
            write(destination / "models" / (key + ".json"), {"theta": result.x.tolist(), "mask": mask,
                  "success": bool(result.success), "loss": float(result.fun), "n_iter": int(result.nit)})
    write(destination / "FIT.json", {"at_utc": now(), "train_ids": [r["case"]["id"] for r in train],
          "model_sha256": {p.name: sha(p) for p in sorted((destination / "models").iterdir())}})
    predictions = {}; audit = {"graphs": 0, "max_graph_error": 0., "gradient_directions": 0,
                               "max_gradient_error": 0., "max_probability_error": 0., "max_feature_error": feature_error}
    for row in rows:
        s = row["sample"]; states = np.asarray(list(product((0, 1), repeat=len(s.action_ids)))); truth = policy_truth(row)
        def save(name, p):
            if min(p) < 0 or abs(p.sum() - 1) > 1e-12: raise ValueError("invalid choice probability")
            slots = []
            for g in s.choice_groups:
                ix = [s.action_ids.index(a) for a in g["action_ids"]]
                invalid = states[:, ix].sum(axis=1) > 1
                if p[invalid].sum() != 0: raise ValueError("illegal multi-choice support")
                q = [float(p[states[:, ix].sum(axis=1) == 0].sum())] + [float(p[states[:, i] == 1].sum()) for i in ix]
                chosen = [i + 1 for i, a in enumerate(g["action_ids"]) if row["label"]["dispatch"][a]]
                chosen = chosen[0] if chosen else 0
                log_loss = -math.log(q[chosen]) if q[chosen] else None
                slots.append({"id": g["id"], "probabilities": q, "chosen": chosen,
                              "correct": int(np.argmax(q) == chosen),
                              "brier": sum((v - int(i == chosen)) ** 2 for i, v in enumerate(q)), "log_loss": log_loss})
            # Deterministic rules can assign zero to actual choices; represent
            # the infinite log loss honestly outside finite model averages.
            predictions.setdefault(name, []).append({"id": row["case"]["id"], "split": row["case"]["split"],
                "priority": json.loads(row["prefix"]["actual_planner_text"])["priority"],
                "sample_sha256": s.sha256, "prefix_sha256": row["prefix_sha256"],
                "joint": p.tolist(), "policy_truth": truth.tolist(), "risk_probability": float(p @ truth),
                "risk_label": row["label"]["risk"], "slots": slots})
        for name, m in models.items():
            w = m._worlds(s, m.theta); p = np.exp(w["log_mass"]).sum(axis=1); save(name, p)
            result = m.predict(s)
            if abs(result["probabilities"]["system:any"] - p @ truth) > 1e-12:
                raise ValueError("independent scoped policy differs")
            expected = reference_worlds(s, row["x"], m.theta[:10])[1] if name.startswith("attribute") else scalar_choice_worlds(s, m.theta[:3])[1]
            error = float(max(abs(expected - p))); audit["max_probability_error"] = max(error, audit["max_probability_error"])
            if error > 1e-12: raise ValueError("independent choice law differs")
            if name.startswith("attribute"):
                for query, probability in result["probabilities"].items():
                    error = abs(evaluate(m.probability_graph(s, output=query)) - probability)
                    audit["graphs"] += 1; audit["max_graph_error"] = max(audit["max_graph_error"], error)
                    if error > 1e-12: raise ValueError("executable policy graph differs")
        for name, theta in standalone.items(): save(name, reference_worlds(s, row["x"], theta)[1])
        bits = [0] * len(s.action_ids)
        priority = json.loads(row["prefix"]["actual_planner_text"])["priority"]
        for g in s.choice_groups:
            selected = next(a for a in g["action_ids"] if a.endswith("enhanced" if priority == "quality" else "basic"))
            bits[s.action_ids.index(selected)] = 1
        p = np.zeros(len(states)); p[next(i for i, v in enumerate(states) if list(v) == bits)] = 1
        save("original_priority_rule", p)
    equivalence = {}
    for penalty in plan["regularization"]:
        m = models[f"attribute_bn_{penalty}"]; theta = standalone[f"standalone_full_{penalty}"]
        probe = m.theta + np.random.default_rng(842).normal(0, .15, len(m.theta))
        grad = m.loss_gradient(probe, [r["sample"] for r in train], [{"dispatch": r["label"]["dispatch"]} for r in train])[1]
        numeric = approx_derivative(lambda t: reference_objective(t[:10], train, penalty)[0] + penalty * (t[10:] @ t[10:]), probe).ravel()
        error = float(max(abs(grad - numeric))); audit["gradient_directions"] += len(probe)
        audit["max_gradient_error"] = max(audit["max_gradient_error"], error)
        if error > 1e-8: raise ValueError("independent objective gradient differs")
        bn_loss = m.loss_gradient(m.theta, [r["sample"] for r in train], [{"dispatch": r["label"]["dispatch"]} for r in train])[0]
        equivalence[str(penalty)] = {"max_coefficient_error": float(max(abs(m.theta[:10] - theta))),
                                    "same_parameter_loss_error": abs(bn_loss - reference_objective(m.theta[:10], train, penalty)[0])}
    write(destination / "PREDICTIONS.json", predictions)
    results = {}
    for name, records in predictions.items():
        results[name] = {}
        for split in ("train", "validation"):
            selected = [r for r in records if r["split"] == split]
            zero = any(s["log_loss"] is None for r in selected for s in r["slots"])
            if zero:
                # Compute finite other metrics but do not silently clip zero.
                safe = json.loads(json.dumps(records))
                for r in safe:
                    for s in r["slots"]:
                        if s["log_loss"] is None: s["log_loss"] = 0.
                result = summarize(safe, split); result["slot_log_loss"] = None
                result["slot_log_loss_status"] = "infinite: positive outcome assigned exact zero probability"
            else: result = summarize(records, split)
            results[name][split] = result
    report = {"finished_at_utc": now(), "plan_sha256": sha(destination / "PLAN.json"),
              "fit_sha256": sha(destination / "FIT.json"), "predictions_sha256": sha(destination / "PREDICTIONS.json"),
              "training_support_sha256": sha(destination / "TRAINING_SUPPORT.json"),
              "methods": len(predictions), "predictions": sum(map(len, predictions.values())),
              "metrics": results, "independent_numerical_checks": audit, "same_feature_logit_equivalence": equivalence,
              "failed_original_flows": failed, "new_api_calls": 0, "new_default_selected": False,
              "scope": plan["evaluation_status"],
              "limits": ["Attributes were designed after inspecting original training aliasing. Neither training improvement nor exposed validation is independent confirmation.",
                         "Only 3/12 original validation flows complete, without balanced or fieldcamp cases. All failures retained.",
                         "Cost-only and warranty-only are feature ablations, not native MAS literature baselines.",
                         "No new unknown shared-planner structure; the added information is also available to the standalone classifier.",
                         "Only a plain unqualified catalogue warranty clause is extracted; not true quality, compliance or causal intent.",
                         "Input is prepared at forecast cutoff, not already delivered. Forecast conditions on unchanged delivery and successful finite command protocol; API/parse failures are not predicted.",
                         "Legacy cue model is explicitly old-feature; attribute variants share independent catalogue evidence and supervision."]}
    write(destination / "REPORT.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "metrics"}), flush=True)
    for name, r in results.items():
        print(json.dumps({"method": name, "train": r["train"], "exposed_validation": r["validation"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("prepare", "run"))
    p.add_argument("--out", type=Path, required=True); p.add_argument("--source-run", type=Path)
    a = p.parse_args()
    if a.mode == "prepare": prepare(a.source_run.resolve(), a.out.resolve())
    else: run(a.out.resolve())
