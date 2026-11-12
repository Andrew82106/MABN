"""Synthetic checks of leakage boundaries, metrics and detector selection only.

No real annotation, learned detector, or held-out score is used by these tests.
"""
import contextlib
from collections import defaultdict
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import evaluate7 as e
import report7


def item(i, group="q", condition="complete", label=0, parse=True):
    return {"item_id": i, "row_id": group+"__"+condition, "question_id": group,
            "group_id": group, "item_index": 1, "condition": condition,
            "parse_ok": parse, "text": "Synthetic assertion.", "split": "test",
            "annotation": {"stance": "asserted", "evidence_relation": "unsupported" if label else "supported",
                           "reference_correctness": "correct", "risk": label}}


def with_scores(rows, by_method):
    for n, row in enumerate(rows):
        row["methods"] = {m: {"score": scores[n], "prediction": int(scores[n] >= .5) if scores[n] is not None else None,
                              "threshold": .5} for m, scores in by_method.items()}
    return rows


class CoreTests(unittest.TestCase):
    def test_multi_claim_behavior_uses_actual_round7_schema(self):
        rows = [item("a"), item("b"), item("c"), item("d")]
        rows[0]["annotation"]["multi_claim"] = True
        rows[1]["annotation"]["multi_claim"] = False
        rows[2]["annotation"]["multiple_facts"] = True
        result = e.behavior(rows)["all"]
        self.assertEqual(result["multiple_facts"], 2)
        self.assertEqual(result["multiple_facts_recorded"], 3)

    def test_vector_threshold_matches_independent_brute_force(self):
        rng = np.random.default_rng(33)
        for _ in range(30):
            y = np.r_[0, 1, rng.integers(0, 2, 28)]
            s = rng.choice([.1, .2, .5, .9], len(y)).astype(float)
            s[-1] = np.nan
            got = e.threshold_search(y, s)
            ok = np.isfinite(s)
            yy, ss = y[ok], s[ok]
            thresholds = [math.nextafter(min(ss), -math.inf), *set(ss), math.nextafter(max(ss), math.inf)]
            result = []
            for t in thresholds:
                p = ss >= t
                tp = sum((yy == 1) & p); fp = sum((yy == 0) & p); fn = int(y.sum())-tp
                f1 = 2*tp/(2*tp+fp+fn)
                precision = tp/(tp+fp) if tp+fp else 0
                result.append((f1, precision, t))
            self.assertEqual(got["threshold"], max(result)[2])
        self.assertIsNone(e.threshold_search([1, 1], [.1, .2])["threshold"])

    def test_missing_positive_is_fn_not_silently_deleted(self):
        rows = with_scores([item("a", label=1, parse=False), item("b", label=0)], {"x": [None, .1]})
        result = e.evaluate_method(rows, "x")
        self.assertEqual(result["all_resolved"]["fn"], 1)
        self.assertEqual(result["scorable_only"]["n"], 1)
        self.assertEqual(result["coverage"]["missing_risk"], 1)
        self.assertEqual(result["top20"]["recall"], 0)
        threshold = e.threshold_search([1, 1, 0], [None, .8, .1])
        self.assertAlmostEqual(threshold["validation_f1"], 2/3)
        self.assertEqual(threshold["n"], 3)
        self.assertEqual(threshold["scorable_n"], 2)

    def test_operational_top20_cannot_prefilter_high_score_abstention(self):
        rows = [item("refusal"), item("risk", label=1), item("safe1"), item("safe2"), item("safe3")]
        rows[0]["annotation"].update(stance="abstained", evidence_relation="not_applicable", risk=None)
        rows = with_scores(rows, {"x": [.99, .9, .3, .2, .1]})
        result = e.evaluate_method(rows, "x")
        primary, oracle = result["top20"], result["top20_resolved_only"]
        self.assertEqual(primary["budget"], 1)
        self.assertEqual(primary["item_ids"], ["refusal"])
        self.assertEqual(primary["selected_abstained"], 1)
        self.assertEqual(primary["selected_resolved"], 0)
        self.assertEqual(primary["confirmed_risk_yield"], 0)
        self.assertEqual(primary["recall"], 0)
        self.assertEqual(oracle["item_ids"], ["risk"])
        self.assertEqual(oracle["precision"], 1)

    def test_operational_top20_budget_and_missing_score_accounting(self):
        rows = [item("unresolved"), item("risk", label=1), item("safe"), item("missing"), item("refusal"), item("risk2", label=1)]
        rows[0]["annotation"].update(evidence_relation="unresolved", risk=None)
        rows[3]["annotation"].update(stance="missing", evidence_relation="not_applicable", risk=None)
        rows[4]["annotation"].update(stance="abstained", evidence_relation="not_applicable", risk=None)
        rows = with_scores(rows, {"x": [.99, .9, None, None, None, None]})
        # A finite score remains usable for ranking even if no threshold produced a prediction.
        rows[0]["methods"]["x"]["prediction"] = None
        primary = e.evaluate_method(rows, "x")["top20"]
        self.assertEqual(primary["budget"], 2)
        self.assertEqual(primary["item_ids"], ["unresolved", "risk"])
        self.assertEqual(primary["selected_unresolved"], 1)
        self.assertEqual(primary["selected_resolved"], 1)
        self.assertEqual(primary["missing_scores"], 4)
        self.assertEqual(primary["missing_predictions"], 5)
        self.assertEqual(primary["confirmed_risk_yield"], .5)
        self.assertEqual(primary["known_risk_total"], 2)
        self.assertEqual(primary["recall"], .5)
        rows[1]["methods"]["x"].update(score=None, prediction=None)
        primary = e.evaluate_method(rows, "x")["top20"]
        self.assertEqual(primary["unfilled_budget"], 1)
        self.assertEqual(primary["selected"], 1)

    def test_pairwise_ranking_ties_and_same_question_conditions(self):
        rows = [item("a", label=1), item("b", label=0),
                item("c", condition="partial", label=1), item("d", condition="partial", label=0)]
        rows = with_scores(rows, {"x": [.9, .1, .5, .5]})
        result = e.evaluate_method(rows, "x")["within_answer"]
        self.assertEqual(result["mixed_responses"], 2)
        self.assertEqual(result["mixed_groups"], 1)
        self.assertEqual(result["mean_response_auc"], .75)

    def test_pearson_direction_and_constant_feature(self):
        y = np.array([0, 0, 1, 1])
        X = np.stack([y, 1-y, np.ones(4)], axis=1)
        risk_order, risk_cor = e.pearson_rank(X, y)
        ecs_order, _ = e.pearson_rank(X, 1-y)
        self.assertEqual(risk_order[0], 0)
        self.assertEqual(ecs_order[0], 1)
        self.assertEqual(risk_cor[2], 0)

    def test_redeep_train_minmax_not_refit_on_test(self):
        model = {"kind": "redeep", "heads": np.array([0]), "layers": np.array([0]),
                 "e_min": 1., "e_range": 2., "p_min": 0., "p_range": 1., "beta": .4}
        values = e.redeep_scores(model, np.array([[1.], [3.], [np.nan]]), np.array([[2.], [0.], [0.]]))
        np.testing.assert_allclose(values[:2], [2., -.4])
        self.assertTrue(np.isnan(values[2]))

    def test_seed_metric_mean_is_not_probability_ensemble(self):
        rows = [item("a", label=1), item("b", label=0)]
        scores = {m: [.9, .1] for m in e.METHODS}
        scores[f"hallurag_mlp_seed_{e.MLP_SEEDS[0]}"] = [.9, .1]
        scores[f"hallurag_mlp_seed_{e.MLP_SEEDS[1]}"] = [.1, .9]
        scores[f"hallurag_mlp_seed_{e.MLP_SEEDS[2]}"] = [.1, .1]
        rows = with_scores(rows, scores)
        result = e.evaluate(rows)
        self.assertAlmostEqual(result["hallurag_mlp_summary"]["all_resolved"]["f1"]["mean"], 1/3)
        self.assertNotIn("hallurag_mlp_mean", rows[0]["methods"])
        boot = e.bootstrap_groups(rows, e.METHODS, draws=30)
        self.assertAlmostEqual(boot["methods"]["hallurag_mlp_mean"]["f1"]["ci95"][0], 1/3)

    def test_group_bootstrap_preserves_paired_items(self):
        rows = [item("a", label=1), item("b", label=0),
                item("c", condition="partial", label=1), item("d", condition="partial", label=0)]
        rows = with_scores(rows, {m: [.9, .1, .9, .1] for m in e.METHODS})
        b = e.bootstrap_groups(rows, e.METHODS, draws=40)
        self.assertEqual(b["groups"], 1)
        self.assertEqual(b["methods"]["hidden_probe"]["f1"]["ci95"], [1., 1.])
        self.assertEqual(b["methods"]["entropy"]["f1_minus_hidden_probe"]["ci95"], [0., 0.])

    def test_mlp_train_scaler_and_seed_determinism(self):
        X = np.array([[0., 1.], [1., 2.], [0., 2.], [1., 1.]], dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        V = X+100
        config = {**e.MLP_CONFIG, "widths": [4, 3, 2], "max_epochs": 3, "patience": 2}
        a = e.train_mlp(X, y, V, y, .001, 17, config)
        b = e.train_mlp(X, y, V, y, .001, 17, config)
        np.testing.assert_allclose(a["scaler"].mean_, X.mean(0))
        self.assertLess(a["scaler"].mean_[0], 2)
        np.testing.assert_array_equal(e.mlp_scores(a, X), e.mlp_scores(b, X))
        self.assertIn(a["training"]["best_epoch"], [1, 2, 3])


class CompleteSyntheticPipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="round7_cpu_test_")
        self.root = Path(self.temp.name).resolve()
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        self.labels = defaultdict(list)
        self.make_fixture()

    def tearDown(self):
        # Check the exact absolute owned temporary target before recursive cleanup.
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        assert self.root.name.startswith("round7_cpu_test_")
        self.temp.cleanup()

    def make_fixture(self):
        root = self.root
        for stage in e.ARRAYS:
            (root/"data"/stage).mkdir(parents=True)
        inputs, generated, baseline = [], [], []
        for split in ["train", "validation", "test", e.EXTERNAL_SPLIT]:
            for q in range(2):
                qid = f"{split}_{q}"
                for condition in ["complete", "partial"]:
                    rid = qid+"__"+condition
                    expected = 1 if split == e.EXTERNAL_SPLIT else 3
                    src = {"row_id": rid, "question_id": qid, "group_id": qid, "split": split,
                           "condition": condition, "dataset": "RAGognize" if split == e.EXTERNAL_SPLIT else "HotpotQA",
                           "expected_items": expected, "questions": ["Synthetic question?"]*expected}
                    row = {**src, "response": "Synthetic fixture only.", "items": []}
                    row_labels = []
                    feature_ids, ys = [], []
                    for index in range(1, expected+1):
                        iid = rid+f"__{index}"
                        risk = int(condition == "partial" and (index == expected or index == 2))
                        parsed = not (split == "test" and q == 1 and condition == "partial" and index == 3)
                        generated_item = {"item_id": iid, "item_index": index, "text": "Synthetic fixture only.",
                                          "parse_ok": parsed, "start": 0, "end": len(row["response"])}
                        row["items"].append(generated_item)
                        row_labels.append({**src, **generated_item, "stance": "asserted", "risk": risk,
                            "evidence_relation": "unsupported" if risk else "supported", "reference_correctness": "correct"})
                        if parsed:
                            feature_ids.append(iid); ys.append(risk)
                            baseline.append({"item_id": iid, "self_risk": .9 if risk else .1, "direct_risk": .8 if risk else .2})
                    n = len(feature_ids)
                    hidden = np.zeros((n, 3584), np.float32)
                    hidden[:, 0] = ys
                    hidden[:, 1] = q*.01
                    heads = np.zeros((n, 784), np.float32); heads[:, 0] = ys
                    ecs = np.zeros((n, 784), np.float32); ecs[:, 0] = 1-np.array(ys)
                    pks = np.zeros((n, 28), np.float32); pks[:, 0] = ys
                    np.savez(root/"data/features"/(rid+".npz"), item_ids=np.array(feature_ids),
                             **{f"hidden_{layer}": hidden for layer in e.LAYERS},
                             surface=hidden[:, :2], surface_names=np.array(["x", "y"]),
                             mean_nll=np.array(ys, np.float32), mean_entropy=np.array(ys, np.float32))
                    np.savez(root/"data/attention"/(rid+".npz"), item_ids=np.array(feature_ids),
                             lookback_features=heads, redeep_ecs=ecs, redeep_pks=pks)
                    np.savez(root/"data/lumina"/(rid+".npz"), item_ids=np.array(feature_ids),
                             lumina_score=np.array(ys, np.float32), lumina_mmd=np.zeros(n), lumina_ipr=np.array(ys)*2)
                    inputs.append(src); generated.append(row)
                    generation_path = root/"data/generation_records"/(rid+".json")
                    e.save(generation_path, row)
                    for annotation in row_labels:
                        annotation["source_generation_sha256"] = e.sha(generation_path)
                        self.labels[split].append(annotation)
        e.savel(root/"data/inputs.jsonl", inputs); e.savel(root/"data/generated.jsonl", generated)
        e.savel(root/"data/baselines.jsonl", baseline)
        for split in ["train", "validation"]:
            e.savel(root/"data"/f"annotations_{split}.jsonl", self.labels[split])
        manifest = {"complete": True, "expected_rows": len(inputs), "completed_rows": len(inputs),
                    "record_files": [r["row_id"]+".json" for r in inputs]}
        for stage in e.ARRAYS:
            e.save(root/"data"/stage/"manifest.json", manifest)
        e.save(root/"data/baselines/manifest.json", manifest)

    def test_bank_accepts_float32_probability_sum_roundoff_without_clipping(self):
        _, _, items = e.metadata(self.root)
        path = self.root/"data/baselines.jsonl"
        baseline = e.readl(path)
        row = baseline[0]
        target = next(i for i in items if i["item_id"] == row["item_id"])
        row_items = [i for i in items if i["row_id"] == target["row_id"]]
        # Same arithmetic as summing two float32 softmax values converted to Python floats.
        over = float(np.float32(.4)) + float(np.float32(.6))
        self.assertEqual(over, 1.0000000298023224)
        self.assertGreater(over, 1.)
        row["direct_risk"] = over
        # A representable float32 excess also checks that Bank does not add clipping.
        row["self_risk"] = float(np.nextafter(np.float32(1.), np.float32(2.)))
        e.savel(path, baseline)
        before = path.read_bytes()
        bank = e.Bank(self.root, row_items)
        self.assertEqual(bank.matrix([target], "direct_risk")[0, 0], np.float32(over))
        self.assertGreater(bank.matrix([target], "self_risk")[0, 0], 1.)
        self.assertEqual(path.read_bytes(), before)

    def test_bank_rejects_substantive_probability_excess_negative_and_nan(self):
        _, _, items = e.metadata(self.root)
        path = self.root/"data/baselines.jsonl"
        baseline = e.readl(path)
        target = next(i for i in items if i["item_id"] == baseline[0]["item_id"])
        row_items = [i for i in items if i["row_id"] == target["row_id"]]
        # Confirm fixture features and row coverage are valid before corrupting probabilities.
        e.Bank(self.root, row_items)
        for key in ("self_risk", "direct_risk"):
            for invalid in (1.01, -1e-8, float("nan"), float("inf"), -float("inf")):
                with self.subTest(key=key, invalid=invalid):
                    baseline[0].update(self_risk=.2, direct_risk=.3)
                    baseline[0][key] = invalid
                    # Deliberately write NaN/Infinity: normal save() sanitizes them to null,
                    # which would test missing-data handling instead of invalid-value rejection.
                    path.write_text("".join(json.dumps(r, allow_nan=True)+"\n" for r in baseline), encoding="utf-8")
                    with self.assertRaises(AssertionError):
                        e.Bank(self.root, row_items)

    def test_all_methods_fit_without_any_held_out_label_file_then_joint_test(self):
        config = {**e.MLP_CONFIG, "widths": [8, 4, 2], "max_epochs": 2, "patience": 1}
        with patch.object(e, "MLP_CONFIG", config), contextlib.redirect_stdout(io.StringIO()):
            e.fit(self.root)
            self.assertFalse((self.root/"data/annotations_test.jsonl").exists())
            self.assertFalse((self.root/"data/annotations_external_test.jsonl").exists())
            frozen = json.loads((self.root/"results/freeze.json").read_text())
            self.assertEqual(set(frozen["method_names"]), set(e.METHODS))
            selection = json.loads((self.root/"results/selection.json").read_text())
            self.assertEqual(len(selection["candidates"]["redeep"]), 288)
            self.assertEqual(len(selection["candidates"]["hidden_probe"]), 12)
            self.assertEqual(len(selection["candidates"]["hallurag_mlp"]), 8)
            self.assertTrue(all(len(c["seeds"]) == 3 for c in selection["candidates"]["hallurag_mlp"]))
            for split in ["test", e.EXTERNAL_SPLIT]:
                e.savel(self.root/"data"/f"annotations_{split}.jsonl", self.labels[split])
            e.test(self.root)
            report7.make_tables(self.root)
        main = json.loads((self.root/"results/metrics_main.json").read_text())
        external = json.loads((self.root/"results/metrics_external.json").read_text())
        self.assertEqual(main["behavior"]["all"]["slots"], 12)
        self.assertEqual(external["behavior"]["all"]["slots"], 4)
        self.assertEqual(main["methods"]["hidden_probe"]["coverage"]["missing_risk"], 1)
        self.assertEqual(main["methods"]["hidden_probe"]["all_resolved"]["fn"], 1)
        self.assertEqual(main["bootstrap"]["draws"], 2000)
        self.assertTrue((self.root/"results/test_complete.json").exists())
        tables = (self.root/"results/TABLES.md").read_text(encoding="utf-8")
        self.assertIn("RAGognize外部留出", tables)
        self.assertIn("非完整原版复现", tables)
        self.assertTrue(all(str(s) in tables for s in e.MLP_SEEDS))
        with self.assertRaises(AssertionError):
            e.test(self.root)

    def test_wrong_split_annotation_is_rejected(self):
        _, _, items = e.metadata(self.root)
        p = self.root/"data/annotations_train.jsonl"
        e.savel(p, self.labels["train"]+[self.labels["test"][0]])
        with self.assertRaises(AssertionError):
            e.load_labels(self.root, items, ["train", "validation"])

    def test_shifted_label_span_is_rejected(self):
        _, _, items = e.metadata(self.root)
        labels = self.labels["train"]
        labels[0]["start"] += 1
        e.savel(self.root/"data/annotations_train.jsonl", labels)
        with self.assertRaisesRegex(AssertionError, "mismatch start"):
            e.load_labels(self.root, items, ["train", "validation"])

    def test_changed_label_text_is_rejected_without_normalizing(self):
        _, _, items = e.metadata(self.root)
        labels = self.labels["train"]
        labels[0]["text"] += " "
        e.savel(self.root/"data/annotations_train.jsonl", labels)
        with self.assertRaisesRegex(AssertionError, "mismatch text"):
            e.load_labels(self.root, items, ["train", "validation"])

    def test_old_label_hash_rejected_after_source_record_changes(self):
        _, _, items = e.metadata(self.root)
        annotation = self.labels["train"][0]
        path = self.root/"data/generation_records"/(annotation["row_id"]+".json")
        generation = json.loads(path.read_text())
        generation["raw_generation_token_ids"] = [4, 9, 12]
        e.save(path, generation)
        with self.assertRaisesRegex(AssertionError, "Stale annotation source_generation_sha256"):
            e.load_labels(self.root, items, ["train", "validation"])

    def test_shared_group_cannot_cross_split(self):
        inputs = e.readl(self.root/"data/inputs.jsonl")
        inputs[-1]["group_id"] = inputs[0]["group_id"]
        e.savel(self.root/"data/inputs.jsonl", inputs)
        with self.assertRaises(AssertionError):
            e.metadata(self.root)

    def test_incomplete_method_blocks_fit_before_any_training(self):
        e.save(self.root/"data/lumina/manifest.json", {"complete": False})
        with self.assertRaises(AssertionError):
            e.fit(self.root)
        self.assertFalse((self.root/"results/freeze.json").exists())


if __name__ == "__main__":
    unittest.main()
