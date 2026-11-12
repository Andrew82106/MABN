"""Read-only supplemental counts after the unified held-out evaluation completes.

Uses saved decisions, never fits a detector or changes thresholds/annotations.
Confusion counts are independently accumulated with stdlib only: this module
does not import evaluate7 or its metrics implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("retained_subject", "missing_subject", "comparison")
METRIC_FIELDS = ("n", "positive", "negative", "tp", "fp", "fn", "tn",
                 "precision", "recall", "f1", "false_positive_rate")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_resolved(record):
    a = record["annotation"]
    return a["stance"] == "asserted" and a["risk"] in (0, 1)


def confusion(records, method):
    """A missing saved prediction means no alert, including FN for a risk item.

    Match the preregistered convention that F1 is undefined when the subset has
    no positive truth items. This is stated explicitly, not silently set to 0.
    """
    cells = Counter(tp=0, fp=0, fn=0, tn=0)
    for record in records:
        if not is_resolved(record):
            continue
        truth = record["annotation"]["risk"]
        prediction = record["methods"][method]["prediction"]
        assert prediction is None or type(prediction) is int and prediction in (0, 1)
        alerted = prediction == 1
        if truth == 1:
            cells["tp" if alerted else "fn"] += 1
        else:
            cells["fp" if alerted else "tn"] += 1
    tp, fp, fn, tn = (cells[k] for k in ("tp", "fp", "fn", "tn"))
    positive, negative = tp + fn, fp + tn
    return {"n": positive + negative, "positive": positive, "negative": negative,
            **cells, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / positive if positive else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if positive else None,
            "false_positive_rate": fp / negative if negative else None}


def assert_metrics(actual, recorded, context):
    for key in METRIC_FIELDS:
        value, expected = actual[key], recorded[key]
        if value is None or expected is None:
            equal = value is expected
        elif key in {"n", "positive", "negative", "tp", "fp", "fn", "tn"}:
            equal = value == expected
        else:
            equal = math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)
        assert equal, f"Independent recount differs at {context}.{key}: {value!r} != {expected!r}"


def partial_role(record, reference):
    """Input-side grouping only; this never supplies an output risk label."""
    if record["condition"] != "partial":
        return None
    index = record["item_index"]
    if "removed_subject_index" in reference:
        removed = reference["removed_subject_index"]
        assert type(removed) is int and removed in (0, 1)
        assert len(reference["subjects"]) == 2 and index in (1, 2, 3)
        if index == 3:
            return "comparison"
        return "missing_subject" if index - 1 == removed else "retained_subject"
    # RAGognize is a single-target control, not a two-subject comparison.
    assert record["split"] == "external_test", "Main data lacks the frozen removed-subject mapping"
    assert index == 1 and len(reference["items"]) == 1
    assert reference["coverage"]["partial"] == [False]
    return "missing_subject"


def subset_report(records, method):
    result = confusion(records, method)
    resolved = [r for r in records if is_resolved(r)]
    normal = [r for r in resolved if r["annotation"]["risk"] == 0]
    result.update(
        all_slots=len(records),
        question_groups=len({r["group_id"] for r in records}),
        supported_normal_items=len(normal),
        risky_asserted_items=sum(r["annotation"]["risk"] == 1 for r in resolved),
        stance_counts=dict(Counter(r["annotation"]["stance"] for r in records)),
        evidence_relation_counts=dict(Counter(r["annotation"]["evidence_relation"] for r in records)),
        scorable_resolved_items=sum(r["methods"][method]["prediction"] is not None for r in resolved),
        missing_predictions_resolved=sum(r["methods"][method]["prediction"] is None for r in resolved),
        missing_predictions_normal=sum(r["methods"][method]["prediction"] is None for r in normal),
        alerts_on_excluded_items=sum(not is_resolved(r) and r["methods"][method]["prediction"] == 1
                                     for r in records),
        false_positive_item_ids=[r["item_id"] for r in normal if r["methods"][method]["prediction"] == 1],
        item_ids=[r["item_id"] for r in records],
    )
    assert result["negative"] == result["supported_normal_items"]
    return result


def validate_predictions(records, methods, frozen, expected_ids, split, references):
    ids = [r["item_id"] for r in records]
    assert len(ids) == len(set(ids)) and ids == expected_ids, "Prediction cohort/order changed"
    for r in records:
        assert r["split"] == split and set(r["methods"]) == set(methods)
        assert r["question_id"] in references and r["condition"] in {"complete", "partial"}
        ref = references[r["question_id"]]
        assert r["item_index"] in [i["item_index"] for i in ref["items"]]
        a = r["annotation"]
        for key in ("item_id", "question_id", "row_id", "split", "condition", "item_index"):
            assert a[key] == r[key], "Embedded annotation identity mismatch: " + key
        relation, stance = a["evidence_relation"], a["stance"]
        assert stance in {"asserted", "abstained", "missing", "tentative"}
        assert relation in {"supported", "unsupported", "contradicted", "unresolved", "not_applicable"}
        truth = int(relation != "supported") if stance == "asserted" and relation in {
            "supported", "unsupported", "contradicted"} else None
        assert a["risk"] == truth, "Inconsistent saved risk label"
        if a.get("false_refusal_rationale") is True:
            assert stance == "asserted" and relation == "contradicted" and truth == 1
        for method in methods:
            saved = r["methods"][method]
            threshold = frozen["thresholds"][method]["threshold"]
            assert saved["threshold"] == threshold, "Saved threshold differs from frozen threshold"
            score = saved["score"]
            usable = score is not None and math.isfinite(score) and threshold is not None and math.isfinite(threshold)
            expected = int(score >= threshold) if usable else None
            assert saved["prediction"] == expected, "Saved prediction disagrees with its frozen threshold"


def evaluate_cohort(records, metrics, methods, references):
    """Recount all stored methods; no method selection or new decision rules."""
    assert set(metrics["methods"]) == set(methods)
    conditions = sorted({r["condition"] for r in records})
    common = [r for r in records if is_resolved(r) and all(
        r["methods"][m]["prediction"] is not None for m in methods)]
    roles = {role: [r for r in records if partial_role(r, references[r["question_id"]]) == role]
             for role in ROLES}
    assert sum(map(len, roles.values())) == sum(r["condition"] == "partial" for r in records)
    flagged = [r for r in records if r["annotation"].get("false_refusal_rationale") is True]
    result = {"all_slots": len(records), "methods": {},
              "role_mapping": {r["item_id"]: role for role, rr in roles.items() for r in rr},
              "false_refusal_rationale": {
                  "selection": "Only the pre-existing boolean annotation flag; no text matching or relabeling",
                  "all_slots": len(flagged), "item_ids": [r["item_id"] for r in flagged],
                  "conditions": dict(Counter(r["condition"] for r in flagged)),
                  "unflagged_or_unspecified_slots": len(records) - len(flagged)},
              "role_applicability": {role: bool(rr) for role, rr in roles.items()}}
    for method in methods:
        stored = metrics["methods"][method]
        all_resolved = confusion(records, method)
        scorable = confusion([r for r in records if r["methods"][method]["prediction"] is not None], method)
        condition_counts = {c: confusion([r for r in records if r["condition"] == c], method) for c in conditions}
        assert_metrics(all_resolved, stored["all_resolved"], method + ".all_resolved")
        assert_metrics(scorable, stored["scorable_only"], method + ".scorable_only")
        assert set(stored["conditions"]) == set(conditions)
        for condition, values in condition_counts.items():
            assert_metrics(values, stored["conditions"][condition], method + "." + condition)
        common_values = confusion(common, method)
        assert_metrics(common_values, metrics["common_scorable"]["methods"][method], method + ".common_scorable")
        assert metrics["common_scorable"]["n"] == len(common)
        result["methods"][method] = {
            "independent_recount": {"all_resolved": all_resolved, "scorable_only": scorable,
                                    "conditions": condition_counts, "common_scorable": common_values,
                                    "matches_original_metrics": True},
            "partial_roles": {role: subset_report(rr, method) for role, rr in roles.items()},
            "false_refusal_rationale": {
                "all": subset_report(flagged, method),
                "conditions": {c: subset_report([r for r in flagged if r["condition"] == c], method)
                               for c in conditions},
                "partial_roles": {role: subset_report([r for r in rr if r["annotation"].get(
                    "false_refusal_rationale") is True], method) for role, rr in roles.items()}}
        }
    return result


def build_supplement(root):
    root = Path(root).resolve()
    out = root / "results"
    complete_path = out / "test_complete.json"
    # Check this before reading any predictions or held-out labels/results.
    if not complete_path.is_file():
        raise RuntimeError("Run the unified frozen test first; results/test_complete.json is required")
    completed = read_json(complete_path)
    assert completed["all_methods_and_both_cohorts_evaluated"] is True
    freeze_path = out / "freeze.json"
    assert sha(freeze_path) == completed["freeze_sha256"]
    frozen = read_json(freeze_path)
    methods = frozen["method_names"]
    assert methods and len(set(methods)) == len(methods)
    reference_path = root / "data/references.jsonl"
    assert sha(reference_path) == frozen["data_sha256"]["data/references.jsonl"]
    reference_rows = read_lines(reference_path)
    references = {r["question_id"]: r for r in reference_rows}
    assert len(references) == len(reference_rows)
    hashes = {"results/test_complete.json": sha(complete_path), "results/freeze.json": sha(freeze_path),
              "data/references.jsonl": sha(reference_path)}
    result = {
        "stage": "post_test_supplement_only", "utc": datetime.now(timezone.utc).isoformat(),
        "method_names": methods, "retuning_or_method_selection": False, "annotation_changes": False,
        "metric_implementation": "Independent stdlib accumulation from saved predictions; evaluate7 not imported",
        "definitions": {
            "normal_denominator": "Actual asserted supported items (risk=0) in each input-defined role; not all retained-subject slots",
            "false_positive_rate": "FP / (FP + TN); null if no actual supported items",
            "missing_prediction": "No alert; resolved positive remains FN and resolved negative remains TN; missing counts also shown",
            "excluded": "Abstained, missing, tentative and unresolved are outside binary metrics; their alerts are counted separately",
            "f1": "2TP / (2TP+FP+FN), null when no positive truth items, matching the frozen protocol",
            "retained_subject": "HotpotQA partial item 1/2 whose zero-based subject index differs from removed_subject_index",
            "missing_subject": "HotpotQA partial item 1/2 for removed_subject_index; external partial uses its sole target with removed evidence",
            "comparison": "HotpotQA partial item 3; external single-target QA has no comparison or retained-subject control",
            "role_is_not_label": "Input-side mapping partitions slots only; output truth always comes from frozen actual-answer annotations",
        }, "cohorts": {}}
    for name, split, item_key in (("main", "test", "test_item_ids"),
                                  ("external", "external_test", "external_item_ids")):
        metric_path = out / f"metrics_{name}.json"
        prediction_path = out / f"predictions_{name}.jsonl"
        assert sha(metric_path) == completed["metrics_sha256"][name]
        metrics = read_json(metric_path)
        assert metrics["split"] == split and metrics["freeze_sha256"] == completed["freeze_sha256"]
        assert metrics["annotation_sha256"] == completed["held_out_annotation_sha256"][split]
        records = read_lines(prediction_path)
        validate_predictions(records, methods, frozen, frozen[item_key], split, references)
        result["cohorts"][name] = evaluate_cohort(records, metrics, methods, references)
        result["cohorts"][name]["split"] = split
        if name == "external":
            assert not result["cohorts"][name]["role_applicability"]["retained_subject"]
            assert not result["cohorts"][name]["role_applicability"]["comparison"]
        hashes[str(metric_path.relative_to(root)).replace("\\", "/")] = sha(metric_path)
        hashes[str(prediction_path.relative_to(root)).replace("\\", "/")] = sha(prediction_path)
    result["input_sha256"] = hashes
    result["script_sha256"] = sha(Path(__file__))
    result["all_independent_metric_assertions_passed"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = build_supplement(args.root)
    path = args.root.resolve() / "results/supplement.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    print("SUPPLEMENT_COMPLETE", path, "sha256=" + sha(path))


if __name__ == "__main__":
    main()
