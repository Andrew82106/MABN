"""Synthetic, CPU-only checks; no formal cohort, predictions or scores opened."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supplement7 import (assert_metrics, build_supplement, confusion, evaluate_cohort,
                         partial_role, subset_report, validate_predictions)


def record(condition, index, truth, prediction, *, stance="asserted", flag=False):
    item_id = f"q__{condition}__{index}"
    relation = "supported" if truth == 0 else "contradicted" if flag else "unsupported"
    if truth is None:
        relation = "unresolved" if stance == "asserted" else "not_applicable"
    r = {"item_id": item_id, "question_id": "q", "row_id": f"q__{condition}",
         "group_id": "q", "split": "test", "condition": condition, "item_index": index,
         "methods": {"detector": {"prediction": prediction, "score": None if prediction is None
                                  else .9 if prediction else .1, "threshold": .5}}}
    r["annotation"] = {k: r[k] for k in ("item_id", "question_id", "row_id", "split", "condition", "item_index")}
    r["annotation"].update(stance=stance, evidence_relation=relation, risk=truth,
                           false_refusal_rationale=flag)
    return r


def manual_metrics(n, positive, negative, tp, fp, fn, tn, precision, recall, f1, fpr):
    # Values are handwritten below, rather than produced by the code under test.
    return dict(n=n, positive=positive, negative=negative, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1, false_positive_rate=fpr)


class SupplementTests(unittest.TestCase):
    def test_missing_predictions_and_nonassertions_stay_separate(self):
        rr = [record("partial", i + 1, truth, pred) for i, (truth, pred) in enumerate(
            [(1, 1), (1, None), (1, 0), (0, 1), (0, 0), (0, None)])]
        rr += [record("partial", 7, None, 1, stance="abstained"),
               record("partial", 8, None, 1),
               record("partial", 9, None, 1, stance="tentative")]
        expected = manual_metrics(6, 3, 3, 1, 1, 2, 2, .5, 1 / 3, .4, 1 / 3)
        self.assertEqual(confusion(rr, "detector"), expected)
        report = subset_report(rr, "detector")
        self.assertEqual(report["all_slots"], 9)
        self.assertEqual(report["supported_normal_items"], 3)
        self.assertEqual(report["missing_predictions_normal"], 1)
        self.assertEqual(report["alerts_on_excluded_items"], 3)
        self.assertEqual(report["false_positive_item_ids"], ["q__partial__4"])

    def test_no_positive_truth_f1_is_undefined_even_with_false_alert(self):
        expected = manual_metrics(1, 0, 1, 0, 1, 0, 0, 0, None, None, 1)
        self.assertEqual(confusion([record("partial", 1, 0, 1)], "detector"), expected)

    def test_role_mapping_does_not_use_actual_risk_label(self):
        ref = {"subjects": ["A", "B"], "removed_subject_index": 0}
        self.assertEqual(partial_role(record("partial", 1, 0, 0), ref), "missing_subject")
        self.assertEqual(partial_role(record("partial", 2, 1, 1), ref), "retained_subject")
        self.assertEqual(partial_role(record("partial", 3, 0, 0), ref), "comparison")
        ref["removed_subject_index"] = 1
        self.assertEqual(partial_role(record("partial", 1, 0, 0), ref), "retained_subject")
        self.assertIsNone(partial_role(record("complete", 1, 0, 0), ref))
        external = record("partial", 1, 0, 0)
        external["split"] = "external_test"
        self.assertEqual(partial_role(external, {"items": [1], "coverage": {"partial": [False]}}),
                         "missing_subject")

    def test_independent_recount_asserts_on_wrong_reported_count(self):
        counts = manual_metrics(4, 2, 2, 1, 1, 1, 1, .5, .5, .5, .5)
        wrong = dict(counts, fp=2)
        with self.assertRaisesRegex(AssertionError, "fp"):
            assert_metrics(counts, wrong, "synthetic")

    def test_hand_counted_cohort_and_frozen_prediction_check(self):
        rr = [record("complete", 1, 0, 0), record("complete", 2, 0, 1),
              record("complete", 3, 1, 0), record("partial", 1, 1, 1, flag=True),
              record("partial", 2, 0, None), record("partial", 3, None, 1, stance="abstained")]
        ref = {"subjects": ["A", "B"], "removed_subject_index": 0,
               "items": [{"item_index": i} for i in (1, 2, 3)]}
        all_resolved = manual_metrics(5, 2, 3, 1, 1, 1, 2, .5, .5, .5, 1 / 3)
        scorable = manual_metrics(4, 2, 2, 1, 1, 1, 1, .5, .5, .5, .5)
        complete = manual_metrics(3, 1, 2, 0, 1, 1, 1, 0, 0, 0, .5)
        partial = manual_metrics(2, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0)
        metrics = {"methods": {"detector": {"all_resolved": all_resolved, "scorable_only": scorable,
                    "conditions": {"complete": complete, "partial": partial}}},
                   "common_scorable": {"n": 4, "methods": {"detector": scorable}}}
        report = evaluate_cohort(rr, metrics, ["detector"], {"q": ref})
        method = report["methods"]["detector"]
        self.assertEqual(method["partial_roles"]["retained_subject"]["tn"], 1)
        self.assertEqual(method["partial_roles"]["retained_subject"]["missing_predictions_normal"], 1)
        self.assertEqual(method["partial_roles"]["comparison"]["n"], 0)
        self.assertEqual(method["false_refusal_rationale"]["all"]["tp"], 1)
        frozen = {"thresholds": {"detector": {"threshold": .5}}}
        validate_predictions(rr, ["detector"], frozen, [r["item_id"] for r in rr], "test", {"q": ref})
        changed = copy.deepcopy(rr)
        changed[0]["methods"]["detector"]["prediction"] = 1
        with self.assertRaisesRegex(AssertionError, "prediction disagrees"):
            validate_predictions(changed, ["detector"], frozen,
                                 [r["item_id"] for r in rr], "test", {"q": ref})

    def test_no_completed_test_means_no_predictions_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            # Invalid JSON would fail differently if this file were opened.
            (root / "results/predictions_main.jsonl").write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "test_complete.json is required"):
                build_supplement(root)
            self.assertFalse((root / "results/supplement.json").exists())


if __name__ == "__main__":
    unittest.main()
