"""Frozen, CPU-only evaluation of the Round 6 evidence-grounding pilot.

Run ``python evaluate6.py fit`` before ``python evaluate6.py test``.
No test label enters fitting, scaling, threshold selection, or model choice.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["hidden21_probe", "surface_logistic", "mean_nll", "mean_entropy",
           "self_confidence", "direct_check", "all_positive", "all_negative"]
SEED = 20260910


def readl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def writej(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def writel(path, values):
    Path(path).write_text("".join(json.dumps(clean(v), ensure_ascii=False,
                                            allow_nan=False) + "\n" for v in values),
                          encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True,
                                      ensure_ascii=False).encode()).hexdigest()


def finite(value):
    return value is not None and np.isfinite(float(value))


def load_inputs(root=ROOT):
    rows = readl(root / "data/generated.jsonl")
    assert len({r["row_id"] for r in rows}) == len(rows), "Duplicate rows"
    items = []
    groups = defaultdict(set)
    for row in rows:
        groups[row["question_id"]].add(row["split"])
        for item in row["items"]:
            items.append({**item, "row_id": row["row_id"],
                          "question_id": row["question_id"], "split": row["split"],
                          "condition": row["condition"]})
    assert all(len(v) == 1 for v in groups.values()), "Question split leakage"
    assert len({i["item_id"] for i in items}) == len(items), "Duplicate items"
    features = np.load(root / "data/features/items.npz", allow_pickle=False)
    feature_ids = [str(x) for x in features["item_ids"]]
    assert len(set(feature_ids)) == len(feature_ids), "Duplicate feature IDs"
    assert features["hidden_21"].shape == (len(feature_ids), 3584), "Unexpected layer21 feature shape"
    assert features["surface"].shape[0] == len(feature_ids)
    assert features["surface"].shape[1] == len(features["surface_names"])
    for key in ["mean_nll", "mean_entropy"]:
        assert features[key].shape == (len(feature_ids),), "Unexpected scalar feature shape " + key
    feature_index = {item_id: n for n, item_id in enumerate(feature_ids)}
    baselines = readl(root / "data/baselines.jsonl")
    assert len({b["item_id"] for b in baselines}) == len(baselines), "Duplicate baselines"
    baseline_index = {b["item_id"]: b for b in baselines}
    item_ids = {i["item_id"] for i in items}
    assert set(feature_index) <= item_ids
    assert set(baseline_index) <= item_ids
    return rows, items, features, feature_index, baseline_index


def annotations_for(root, allowed_ids):
    """Only retain annotations belonging to the requested split's item IDs."""
    annotations = {}
    with (root / "data/annotations.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            annotation = json.loads(line)
            item_id = annotation["item_id"]
            if item_id not in allowed_ids:
                continue
            assert item_id not in annotations, "Duplicate annotation " + item_id
            stance = annotation["stance"]
            relation = annotation["evidence_relation"]
            expected = (0 if relation == "supported" else 1) if (
                stance == "asserted" and relation in {
                    "supported", "unsupported", "contradicted"}) else None
            assert annotation.get("risk") == expected, "Inconsistent risk label " + item_id
            annotations[item_id] = annotation
    assert set(annotations) == set(allowed_ids), "Missing annotations: " + str(
        sorted(set(allowed_ids) - set(annotations)))
    return annotations


def resolved(annotation):
    return annotation.get("risk") in (0, 1) and annotation["stance"] == "asserted"


def binary_metrics(labels, predictions, scores=None):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    positive, negative = tp + fn, fp + tn
    reasons = {}
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positive if positive else None
    f1 = 2 * tp / (2 * tp + fp + fn) if positive else None
    fpr = fp / negative if negative else None
    if precision is None:
        reasons["precision"] = "No predicted positives"
    if not positive:
        reasons["recall"] = reasons["f1"] = "No positive reference items"
    if not negative:
        reasons["false_positive_rate"] = "No negative reference items"
    auc = ap = None
    ranking_n = 0
    if scores is not None:
        available = np.array([finite(s) for s in scores], dtype=bool)
        ranking_labels = labels[available]
        ranking_scores = np.array([float(s) for s in scores if finite(s)])
        ranking_n = len(ranking_labels)
        if len(np.unique(ranking_labels)) == 2:
            auc = float(roc_auc_score(ranking_labels, ranking_scores))
        else:
            reasons["auroc"] = "Scorable subset lacks both reference classes"
        if ranking_labels.sum() > 0:
            ap = float(average_precision_score(ranking_labels, ranking_scores))
        else:
            reasons["average_precision"] = "Scorable subset has no positive reference items"
    return {"n": len(labels), "positive": positive, "negative": negative,
            "prevalence": positive / len(labels) if len(labels) else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": fpr, "auroc": auc,
            "average_precision": ap, "ranking_scorable_n": ranking_n,
            "undefined_reasons": reasons}


def select_threshold(labels, scores):
    available = [n for n, score in enumerate(scores) if finite(score)]
    y = np.array([labels[n] for n in available], dtype=int)
    s = np.array([scores[n] for n in available], dtype=float)
    if not len(s) or len(np.unique(y)) != 2:
        return {"status": "unavailable", "threshold": None,
                "reason": "Validation scorable items must contain both reference classes",
                "validation_scorable_n": len(s)}
    # Endpoints represent all-alert and no-alert. Strictly greater than maximum
    # is finite so the frozen JSON never silently turns infinity into null.
    candidates = [float(np.nextafter(s.min(), -np.inf))]
    candidates.extend(float(v) for v in np.unique(s))
    candidates.append(float(np.nextafter(s.max(), np.inf)))
    ranked = []
    for threshold in candidates:
        m = binary_metrics(y, s >= threshold, s)
        ranked.append(((m["f1"], m["precision"] or 0.0, threshold), threshold, m))
    _, threshold, metrics = max(ranked, key=lambda x: x[0])
    return {"status": "ready", "threshold": threshold,
            "comparison": ">=", "candidate_count": len(candidates),
            "tie_break": "risk F1, precision (undefined treated as 0), higher threshold",
            "validation_scorable_n": len(s), "selected_metrics": metrics}


def feature_vector(item, features, index, key):
    if not item.get("parse_ok", False) or item["item_id"] not in index:
        return None
    vector = np.asarray(features[key][index[item["item_id"]]], dtype=float)
    return vector if np.isfinite(vector).all() else None


def train_classifier(items, annotations, features, index, key):
    pairs = [(i, feature_vector(i, features, index, key)) for i in items
             if i["split"] == "train" and resolved(annotations[i["item_id"]])]
    pairs = [(i, vector) for i, vector in pairs if vector is not None]
    y = np.asarray([annotations[i["item_id"]]["risk"] for i, _ in pairs])
    manifest = {"feature": key, "train_item_ids": [i["item_id"] for i, _ in pairs],
                "n_train": len(pairs), "classes": dict(Counter(int(x) for x in y)),
                "parameters": {"C": .1, "class_weight": "balanced", "solver": "liblinear",
                               "penalty": "l2", "max_iter": 2000, "random_state": SEED}}
    if len(np.unique(y)) < 2:
        return None, {**manifest, "status": "unavailable", "reason": "Train lacks both classes"}
    X = np.stack([vector for _, vector in pairs])
    classifier = make_pipeline(StandardScaler(), LogisticRegression(
        C=.1, class_weight="balanced", solver="liblinear", penalty="l2",
        max_iter=2000, random_state=SEED))
    classifier.fit(X, y)
    manifest.update(status="ready", feature_width=X.shape[1],
                    iterations=classifier[-1].n_iter_.tolist(),
                    reached_iteration_limit=bool((classifier[-1].n_iter_ >= 2000).any()))
    return classifier, manifest


def score_item(item, features, index, baselines, classifiers):
    scores = {method: None for method in METHODS}
    reasons = {}
    # Label-free constants also flag items whose automatic boundary failed.
    scores["all_positive"] = 1.0
    scores["all_negative"] = 0.0
    if not item.get("parse_ok", False):
        for method in METHODS[:-2]:
            reasons[method] = "automatic_item_parse_failed"
        return scores, reasons
    for method, key in [("hidden21_probe", "hidden_21"), ("surface_logistic", "surface")]:
        vector = feature_vector(item, features, index, key)
        classifier = classifiers.get(method)
        if vector is not None and classifier is not None:
            scores[method] = float(classifier.predict_proba(vector[None])[0, 1])
        else:
            reasons[method] = "feature_or_trained_classifier_unavailable"
    n = index.get(item["item_id"])
    for method in ["mean_nll", "mean_entropy"]:
        if n is not None and finite(features[method][n]):
            scores[method] = float(features[method][n])
        else:
            reasons[method] = "feature_unavailable"
    baseline = baselines.get(item["item_id"], {})
    confidence = baseline.get("self_confidence")
    if finite(confidence) and 0 <= float(confidence) <= 100:
        scores["self_confidence"] = 1.0 - float(confidence) / 100.0
    else:
        reasons["self_confidence"] = "self_confidence_unparseable_or_missing"
    direct = baseline.get("direct_risk")
    if finite(direct) and 0 <= float(direct) <= 1:
        probs = baseline.get("direct_probs")
        if probs is not None:
            assert len(probs) == 3 and np.allclose(sum(probs), 1, atol=1e-5)
            assert np.isclose(direct, probs[1] + probs[2], atol=1e-5)
        scores["direct_check"] = float(direct)
    else:
        reasons["direct_check"] = "direct_check_missing_or_nonfinite"
    return scores, reasons


def make_predictions(items, annotations, features, index, baselines, classifiers, threshold_info):
    predictions = []
    for item in items:
        scores, reasons = score_item(item, features, index, baselines, classifiers)
        methods = {}
        for method in METHODS:
            score = scores[method]
            info = threshold_info[method]
            threshold = info.get("threshold")
            pred = int(score >= threshold) if finite(score) and finite(threshold) else None
            methods[method] = {"score": score, "threshold": threshold, "prediction": pred,
                               "missing_reason": reasons.get(method) or (
                                   info.get("reason") if pred is None else None)}
        predictions.append({**item, "annotation": annotations[item["item_id"]], "methods": methods})
    return predictions


def summarize_method(records, method):
    cohort = [p for p in records if resolved(p["annotation"])]
    scorable = [p for p in cohort if p["methods"][method]["prediction"] is not None]
    missing = [p for p in cohort if p["methods"][method]["prediction"] is None]
    def measure(part):
        return binary_metrics([p["annotation"]["risk"] for p in part],
                              [p["methods"][method]["prediction"] or 0 for p in part],
                              [p["methods"][method]["score"] for p in part])
    result = {"end_to_end": measure(cohort), "scorable_only": measure(scorable),
              "coverage": {"resolved_items": len(cohort), "scorable_resolved_items": len(scorable),
                           "resolved_fraction": len(scorable) / len(cohort) if cohort else None,
                           "missing_resolved_items": len(missing),
                           "missing_risk_items": sum(p["annotation"]["risk"] for p in missing),
                           "missing_reasons": dict(Counter(p["methods"][method]["missing_reason"] for p in missing)),
                           "all_slots": len(records),
                           "scorable_all_slots": sum(p["methods"][method]["prediction"] is not None for p in records)},
              "by_condition": {condition: measure([p for p in cohort if p["condition"] == condition])
                               for condition in sorted({p["condition"] for p in records})}}
    subtype = {}
    filters = {"unsupported_reference_correct": lambda a: a["evidence_relation"] == "unsupported" and a["reference_correctness"] == "correct",
               "unsupported_reference_incorrect": lambda a: a["evidence_relation"] == "unsupported" and a["reference_correctness"] == "incorrect",
               "unsupported_reference_unresolved": lambda a: a["evidence_relation"] == "unsupported" and a["reference_correctness"] == "unresolved",
               "contradicted": lambda a: a["evidence_relation"] == "contradicted"}
    for name, condition in filters.items():
        part = [p for p in cohort if condition(p["annotation"])]
        detected = sum(p["methods"][method]["prediction"] == 1 for p in part)
        subtype[name] = {"n": len(part), "detected": detected,
                         "recall": detected / len(part) if part else None}
    result["risk_subtypes"] = subtype
    # Fixed lexical item-ID order breaks ties; no test-label-dependent selection.
    budget = math.ceil(.2 * len(cohort))
    ranked = sorted(scorable, key=lambda p: (-p["methods"][method]["score"], p["item_id"]))
    selected = ranked[:budget]
    found = sum(p["annotation"]["risk"] for p in selected)
    positive = sum(p["annotation"]["risk"] for p in cohort)
    result["top20_review"] = {"denominator": "all resolved asserted items", "requested_budget": budget,
                              "reviewed": len(selected), "risk_found": found,
                              "supported_reviewed": len(selected) - found,
                              "precision": found / len(selected) if selected else None,
                              "recall": found / positive if positive else None,
                              "item_ids": [p["item_id"] for p in selected]}
    by_answer = defaultdict(list)
    for p in cohort:
        by_answer[p["row_id"]].append(p)
    mixed = {row_id: part for row_id, part in by_answer.items()
             if len({p["annotation"]["risk"] for p in part}) == 2}
    answer_aucs = []
    possible_pairs = scored_pairs = 0
    correct_pairs = 0.0
    for row_id, part in mixed.items():
        good = [p for p in part if p["methods"][method]["prediction"] is not None]
        if len({p["annotation"]["risk"] for p in good}) == 2:
            auc = roc_auc_score([p["annotation"]["risk"] for p in good],
                                [p["methods"][method]["score"] for p in good])
            answer_aucs.append({"row_id": row_id, "auroc": float(auc)})
        for pos in [p for p in part if p["annotation"]["risk"] == 1]:
            for neg in [p for p in part if p["annotation"]["risk"] == 0]:
                possible_pairs += 1
                a, b = pos["methods"][method], neg["methods"][method]
                if a["prediction"] is not None and b["prediction"] is not None:
                    scored_pairs += 1
                    correct_pairs += 1 if a["score"] > b["score"] else .5 if a["score"] == b["score"] else 0
    result["within_mixed_answer"] = {
        "mixed_answers": len(mixed), "scorable_mixed_answers": len(answer_aucs),
        "mean_answer_auroc": float(np.mean([v["auroc"] for v in answer_aucs])) if answer_aucs else None,
        "possible_pairs": possible_pairs, "scored_pairs": scored_pairs,
        "paired_ranking": correct_pairs / scored_pairs if scored_pairs else None,
        "per_answer": answer_aucs}
    # Within-question matched facts that stay supported in both conditions.
    paired = defaultdict(dict)
    for p in cohort:
        paired[(p["question_id"], p["item_index"])][p["condition"]] = p
    controls = []
    for (qid, item_index), pair in paired.items():
        if len(pair) != 2 or any(p["annotation"]["risk"] != 0 for p in pair.values()):
            continue
        complete = pair.get("complete")
        partial = next((p for cond, p in pair.items() if cond != "complete"), None)
        if complete is None or partial is None:
            continue
        a, b = complete["methods"][method], partial["methods"][method]
        if finite(a["score"]) and finite(b["score"]):
            controls.append({"question_id": qid, "item_index": item_index,
                             "complete_score": a["score"], "partial_score": b["score"],
                             "delta": b["score"] - a["score"],
                             "complete_alert": a["prediction"], "partial_alert": b["prediction"]})
    result["both_conditions_supported_control"] = {
        "n": len(controls), "mean_partial_minus_complete": float(np.mean([c["delta"] for c in controls])) if controls else None,
        "pairs": controls,
        "caveat": "Aligned target items; generated wording can differ. Not a causal mechanism proof."}
    return result


def behavior_summary(records):
    def one(part):
        return {"slots": len(part), "answers": len({p["row_id"] for p in part}),
                "question_groups": len({p["question_id"] for p in part}),
                "stance": dict(Counter(p["annotation"]["stance"] for p in part)),
                "evidence_relation": dict(Counter(p["annotation"]["evidence_relation"] for p in part)),
                "reference_correctness": dict(Counter(p["annotation"]["reference_correctness"] for p in part)),
                "joint_labels": dict(Counter(" / ".join([p["annotation"]["stance"],
                    p["annotation"]["evidence_relation"], p["annotation"]["reference_correctness"]]) for p in part)),
                "resolved_asserted": sum(resolved(p["annotation"]) for p in part),
                "risk_items": sum(p["annotation"].get("risk") == 1 for p in part),
                "parse_failures": sum(not p.get("parse_ok", False) for p in part),
                "multiple_fact_items": sum(bool(p["annotation"].get("multiple_facts", False)) for p in part),
                "multi_fact_field_recorded": sum("multiple_facts" in p["annotation"] for p in part),
                "annotators": dict(Counter(p["annotation"].get("annotator", "unrecorded") for p in part))}
    return {"overall": one(records),
            "by_condition": {condition: one([p for p in records if p["condition"] == condition])
                             for condition in sorted({p["condition"] for p in records})}}


def evaluate_records(records):
    common = [p for p in records if resolved(p["annotation"]) and all(
        p["methods"][method]["prediction"] is not None for method in METHODS)]
    return {"behavior": behavior_summary(records),
            "methods": {method: summarize_method(records, method) for method in METHODS},
            "common_scorable_comparison": {"n": len(common), "item_ids": [p["item_id"] for p in common],
                "methods": {method: binary_metrics([p["annotation"]["risk"] for p in common],
                    [p["methods"][method]["prediction"] for p in common],
                    [p["methods"][method]["score"] for p in common]) for method in METHODS}}}


def input_hashes(root):
    files = ["data/generated.jsonl", "data/features/items.npz", "data/baselines.jsonl",
             "design_v1.json", "prompts/answer.txt"]
    for optional in ["protocol.json", "data/rows.jsonl", "data/questions.jsonl",
                     "data/inputs.jsonl", "data/references.jsonl"]:
        if (root / optional).exists():
            files.append(optional)
    return {file: sha(root / file) for file in files}


def fit(root):
    out = root / "results"
    out.mkdir(exist_ok=True)
    assert not (out / "freeze.json").exists(), "Already frozen; do not overwrite a frozen experiment"
    _, items, features, index, baselines = load_inputs(root)
    development = [i for i in items if i["split"] in {"train", "validation"}]
    annotations = annotations_for(root, {i["item_id"] for i in development})
    classifiers, training = {}, {}
    for method, key in [("hidden21_probe", "hidden_21"), ("surface_logistic", "surface")]:
        classifier, info = train_classifier(development, annotations, features, index, key)
        classifiers[method], training[method] = classifier, info
    model_path = out / "classifiers.pkl"
    model_path.write_bytes(pickle.dumps(classifiers, protocol=pickle.HIGHEST_PROTOCOL))
    validation = [i for i in development if i["split"] == "validation"]
    valid_resolved = [i for i in validation if resolved(annotations[i["item_id"]])]
    raw = {i["item_id"]: score_item(i, features, index, baselines, classifiers)[0] for i in valid_resolved}
    thresholds = {}
    for method in METHODS:
        if method in {"all_positive", "all_negative"}:
            thresholds[method] = {"status": "ready", "threshold": .5,
                                  "reason": "Fixed untrained constant baseline; no threshold optimization"}
        else:
            thresholds[method] = select_threshold(
                [annotations[i["item_id"]]["risk"] for i in valid_resolved],
                [raw[i["item_id"]][method] for i in valid_resolved])
    predictions = make_predictions(development, annotations, features, index, baselines, classifiers, thresholds)
    writel(out / "development_predictions.jsonl", predictions)
    validation_metrics = evaluate_records([p for p in predictions if p["split"] == "validation"])
    writej(out / "validation.json", validation_metrics)
    ordered_annotations = [annotations[item_id] for item_id in sorted(annotations)]
    writej(out / "training_manifest.json", {"models": training, "surface_names": features["surface_names"].tolist(),
                                            "development_annotation_sha256": object_sha(ordered_annotations)})
    frozen = {"stage": "fit_complete_test_not_evaluated", "utc": datetime.now(timezone.utc).isoformat(),
              "primary_method": "hidden21_probe", "unit": "resolved asserted answer item",
              "primary_counting": "All resolved asserted items; missing predictions produce no alert and risky misses count as FN. Scorable-only is separate.",
              "ranking_note": "AUROC and average precision use available scores only, with coverage reported. PR-AUC is average precision, not trapezoidal integration.",
              "training": training, "thresholds": thresholds, "input_sha256": input_hashes(root),
              "model_sha256": sha(model_path), "development_annotation_sha256": object_sha(ordered_annotations),
              "development_item_ids": sorted(annotations),
              "test_item_ids": sorted(i["item_id"] for i in items if i["split"] == "test"),
              "evaluator_sha256": sha(Path(__file__)),
              "test_used_for_fitting_or_threshold_selection": False,
              "pilot_inference_limits": "30 curated question groups; six held-out groups. Assistant evidence annotations, not independent human gold. No robust 0.7 or mechanism claim."}
    writej(out / "freeze.json", frozen)
    print(json.dumps({"fit": "complete", "train_n": sum(i["split"] == "train" for i in development),
                      "validation_n": len(validation), "thresholds": thresholds}, ensure_ascii=False), flush=True)


def test(root):
    out = root / "results"
    frozen = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    assert not (out / "metrics.json").exists(), "Test was already evaluated; do not overwrite"
    assert frozen["input_sha256"] == input_hashes(root), "Frozen input changed"
    assert frozen["evaluator_sha256"] == sha(Path(__file__)), "Evaluator changed after freeze"
    assert frozen["model_sha256"] == sha(out / "classifiers.pkl"), "Frozen classifier changed"
    development_annotations = annotations_for(root, set(frozen["development_item_ids"]))
    assert object_sha([development_annotations[k] for k in sorted(development_annotations)]) == frozen["development_annotation_sha256"], "Development annotations changed"
    _, items, features, index, baselines = load_inputs(root)
    test_items = [i for i in items if i["split"] == "test"]
    assert sorted(i["item_id"] for i in test_items) == frozen["test_item_ids"]
    annotations = annotations_for(root, set(frozen["test_item_ids"]))
    classifiers = pickle.loads((out / "classifiers.pkl").read_bytes())
    predictions = make_predictions(test_items, annotations, features, index, baselines,
                                   classifiers, frozen["thresholds"])
    writel(out / "predictions.jsonl", predictions)
    metrics = evaluate_records(predictions)
    metrics.update(stage="frozen_held_out_pilot_evaluated", utc=datetime.now(timezone.utc).isoformat(),
                   primary_method=frozen["primary_method"], freeze_sha256=sha(out / "freeze.json"),
                   test_annotation_sha256=object_sha([annotations[k] for k in sorted(annotations)]),
                   notes=["Fact-item F1, not token or whole-response F1",
                          "Missing predictions count as no alert in end-to-end; their counts remain explicit",
                          "Available-score ranking metrics are not substitutes for F1",
                          "Only six independent held-out question groups; do not claim stable generalization"])
    writej(out / "metrics.json", metrics)
    print(json.dumps({method: metrics["methods"][method]["end_to_end"] for method in METHODS}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "test"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    (fit if args.stage == "fit" else test)(args.root.resolve())


if __name__ == "__main__":
    main()
