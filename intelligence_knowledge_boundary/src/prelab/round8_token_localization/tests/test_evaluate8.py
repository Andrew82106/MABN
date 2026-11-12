"""Synthetic counting/label-integrity checks. No real token scores or model fitting."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import evaluate8 as e


def fixture(text, offsets, specs, row="q__complete"):
    """spec=(start,end,original risk,status,error intervals[,stance])."""
    generated = {"row_id": row, "question_id": row.split("__")[0], "response": text,
                 "response_token_ids": list(range(len(offsets))), "response_token_offsets": offsets}
    items, gold, old = [], {}, {}
    for index, spec in enumerate(specs, 1):
        start, end, risk, status, spans = spec[:5]
        stance = spec[5] if len(spec) > 5 else "asserted"
        iid = row+f"__{index}"
        item = {"item_id": iid, "row_id": row, "question_id": generated["question_id"], "group_id": generated["question_id"],
                "split": "validation", "condition": "complete", "text": text[start:end], "start": start, "end": end}
        items.append(item)
        gold[iid] = {**item, "source_generation_sha256": "abc", "original_risk": risk, "original_stance": stance,
                     "localization_status": status, "risk_spans": [{"start": a, "end": b, "text": text[a:b]} for a, b in spans]}
        old[iid] = {"annotation": {"risk": risk, "stance": stance}}
    return generated, items, gold, old


def align_score(f, scores):
    tokens, spans = e.align_row(*f)
    for token, score in zip(tokens, scores):
        token["scores"]["x"] = score
    return tokens, spans


class CountingTests(unittest.TestCase):
    def test_numeric_bpe_pieces_and_function_word_retained_punctuation_excluded(self):
        f = fixture("Born 1908.", [(0, 4), (4, 6), (6, 7), (7, 9), (9, 10)],
                    [(0, 10, 1, "resolved", [(5, 9)])])
        tokens, spans = align_score(f, [.1, .9, .9, .9, .9])
        self.assertEqual([t["gold"] for t in tokens], [0, 1, 1, 1, None])
        m = e.localization_metrics(tokens, spans, "x", .5)
        self.assertEqual((m["micro"]["tp"], m["micro"]["tn"], m["micro"]["tokens"]), (3, 1, 4))
        self.assertEqual(m["span_regions"]["mean_token_coverage"], 1)

    def test_cross_item_token_is_one_positive_not_positive_and_negative(self):
        f = fixture("ab", [(0, 2)], [(0, 1, 0, "resolved", []), (1, 2, 1, "resolved", [(1, 2)])])
        tokens, spans = align_score(f, [.9])
        m = e.localization_metrics(tokens, spans, "x", .5)
        self.assertEqual((m["micro"]["tp"], m["micro"]["tn"], m["micro"]["tokens"]), (1, 0, 1))
        self.assertEqual(m["character_coverage"]["precision"], .5)
        self.assertEqual(m["span_regions"]["any_hit_recall"], 1)

    def test_unknown_cross_boundary_excludes_whole_token_without_false_normal_label(self):
        f = fixture("ab", [(0, 2)], [(0, 1, 0, "resolved", []), (1, 2, 1, "unresolved", [])])
        tokens, spans = align_score(f, [.9])
        self.assertFalse(tokens[0]["main_eligible"])
        self.assertIsNone(tokens[0]["gold"])
        self.assertEqual(e.localization_metrics(tokens, spans, "x", .5)["micro"]["tokens"], 0)

    def test_duplicate_overlapping_adjacent_spans_merge_before_region_counting(self):
        a = fixture("abcde", [(n, n+1) for n in range(5)], [(0, 5, 1, "resolved", [(1, 3), (1, 3), (2, 4), (4, 5)])])
        b = fixture("abcde", [(n, n+1) for n in range(5)], [(0, 5, 1, "resolved", [(1, 5)])])
        ta, sa = align_score(a, [.1, .9, .1, .1, .1]); tb, sb = align_score(b, [.1, .9, .1, .1, .1])
        self.assertEqual([r["gold"] for r in ta], [r["gold"] for r in tb])
        ma = e.localization_metrics(ta, sa, "x", .5); mb = e.localization_metrics(tb, sb, "x", .5)
        self.assertEqual(ma["span_regions"], mb["span_regions"])
        self.assertEqual(ma["span_regions"]["n"], 1)
        self.assertEqual(ma["span_regions"]["mean_character_coverage"], .25)

    def test_within_risky_items_keeps_nonspan_tokens_and_shows_broadcast_redundancy(self):
        f = fixture("old 1908", [(0, 3), (4, 8)], [(0, 8, 1, "resolved", [(4, 8)])])
        tokens, spans = align_score(f, [.9, .9])
        m = e.localization_metrics(tokens, spans, "x", .5, risk_only=True)
        self.assertEqual((m["micro"]["tp"], m["micro"]["fp"]), (1, 1))
        self.assertAlmostEqual(m["micro"]["f1"], 2/3)
        self.assertEqual(m["span_regions"]["any_hit_recall"], 1)
        self.assertEqual(m["character_coverage"]["extra_alert_characters"], 3)

    def test_normal_answer_false_positive_counts_micro_not_fake_macro_f1(self):
        risky = fixture("ab", [(0, 1), (1, 2)], [(0, 2, 1, "resolved", [(1, 2)])], row="r__complete")
        normal = fixture("cd", [(0, 1), (1, 2)], [(0, 2, 0, "resolved", [])], row="n__complete")
        tr, sr = align_score(risky, [.1, .1]); tn, sn = align_score(normal, [.9, .1])
        m = e.localization_metrics(tr+tn, sr+sn, "x", .5)
        self.assertEqual((m["micro"]["fp"], m["micro"]["fn"], m["micro"]["tokens"]), (1, 1, 4))
        self.assertEqual(m["macro_by_answer"]["f1"], {"mean": 0., "defined_answers": 1})
        self.assertEqual(m["fully_normal_answers"]["any_false_alert_rate"], 1.)
        self.assertEqual(m["fully_normal_answers"]["token_false_positive_rate"], .5)

    def test_partly_unlocalized_answer_is_not_fully_normal(self):
        f = fixture("ab cd", [(0, 2), (3, 5)], [(0, 2, 0, "resolved", []), (3, 5, 1, "unresolved", [])])
        tokens, spans = align_score(f, [.1, .9])
        m = e.localization_metrics(tokens, spans, "x", .5)
        self.assertEqual(m["micro"]["tokens"], 1)
        self.assertEqual(m["fully_normal_answers"]["n"], 0)
        self.assertIsNone(m["micro"]["auroc"])

    def test_missing_score_keeps_gold_positive_as_false_negative(self):
        f = fixture("ab", [(0, 1), (1, 2)], [(0, 2, 1, "resolved", [(1, 2)])])
        tokens, spans = align_score(f, [.1, None])
        m = e.localization_metrics(tokens, spans, "x", .5)
        self.assertEqual((m["micro"]["fn"], m["micro"]["missing_predictions"], m["micro"]["tokens"]), (1, 1, 2))
        self.assertEqual(m["span_regions"]["any_hit_recall"], 0)

    def test_threshold_maximizes_validation_f1_with_conservative_ties(self):
        rows = [{"gold": y, "scores": {"x": s}} for y, s in [(1, .8), (0, .7), (1, .2), (0, .1), (1, None)]]
        got = e.choose_threshold(rows, "x")
        candidates = [-1., .1, .2, .7, .8, 1.]
        ranked = []
        for threshold in candidates:
            m = e.confusion(rows, "x", threshold)
            ranked.append((m["f1"], m["precision"] or 0., threshold))
        self.assertEqual(got["threshold"], max(ranked)[2])
        self.assertAlmostEqual(got["validation_f1"], 2/3)
        self.assertIsNone(e.choose_threshold([{"gold": 1, "scores": {"x": .5}}], "x")["threshold"])

    def test_refusal_alert_is_descriptive_and_does_not_enter_main(self):
        f = fixture("unknown", [(0, 7)], [(0, 7, None, "excluded", [], "abstained")])
        tokens, spans = align_score(f, [.9])
        self.assertEqual(e.localization_metrics(tokens, spans, "x", .5)["micro"]["tokens"], 0)
        self.assertEqual(e.describe_alerts(tokens, "x", .5)["refusal_text"]["alerts"], 1)

    def test_group_bootstrap_keeps_paired_conditions_and_own_broadcast(self):
        a = fixture("ab", [(0, 1), (1, 2)], [(0, 2, 1, "resolved", [(1, 2)])], row="q__complete")
        b = fixture("cd", [(0, 1), (1, 2)], [(0, 2, 1, "resolved", [(1, 2)])], row="q__partial")
        # Opposite condition-level outcomes: independent answer resampling would
        # vary F1, whereas sampling the whole sole group must keep it fixed.
        ta, sa = align_score(a, [.1, .9]); tb, sb = align_score(b, [.9, .1])
        for t in ta+tb:
            t["scores"] = {"x__token": t["scores"]["x"], "x__broadcast": .9}
        thresholds = {m: {"threshold": .5} for m in ["x__token", "x__broadcast"]}
        metrics = e.evaluate(ta+tb, sa+sb, thresholds)["methods"]
        boot = e.group_bootstrap(ta+tb, list(thresholds), metrics, draws=30)
        self.assertEqual(boot["groups"], 1)
        for subset in ["all_resolved_items", "risk_items_only"]:
            result = boot["subsets"][subset]["x__token"]
            self.assertEqual(result["micro_f1"]["ci95"], [.5, .5])
            np.testing.assert_allclose(result["micro_f1_minus_own_broadcast"]["ci95"], [-1/6, -1/6])

    def test_three_seed_summary_averages_metrics_not_scores(self):
        f = fixture("ab", [(0, 1), (1, 2)], [(0, 2, 1, "resolved", [(1, 2)])])
        tokens, spans = align_score(f, [.1, .9])
        patterns = ([.1, .9], [.9, .1], [.1, .1])
        names = [f"hallurag_mlp_seed_{s}__token" for s in e.MLP_SEEDS]
        for n, token in enumerate(tokens):
            token["scores"] = {name: score[n] for name, score in zip(names, patterns)}
        result = e.evaluate(tokens, spans, {name: {"threshold": .5} for name in names})
        self.assertAlmostEqual(result["mlp_seed_summary"]["token"]["subsets"]["all_resolved_items"]["f1"]["mean"], 1/3)
        self.assertEqual(set(result["methods"]), set(names))


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="round8_count_test_")
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        assert self.root.parent == Path(tempfile.gettempdir()).resolve()
        assert self.root.name.startswith("round8_count_test_")
        self.temp.cleanup()

    def test_span_hash_text_and_empty_risk_location_fail_closed(self):
        gen, items, gold, old = fixture("1908", [(0, 4)], [(0, 4, 1, "resolved", [(0, 4)])])
        path = self.root/"spans_validation.jsonl"
        e.savel(path, list(gold.values()))
        records = {gen["row_id"]: (gen, "abc")}
        self.assertEqual(len(e.load_gold(path, items, old, records)), 1)
        a = next(iter(gold.values()))
        for key, value in [("source_generation_sha256", "old"), ("text", "1909"), ("risk_spans", [])]:
            changed = {**a, key: value}
            e.savel(path, [changed])
            with self.assertRaises(AssertionError):
                e.load_gold(path, items, old, records)
        shifted = {**a, "risk_spans": [{"start": 1, "end": 4, "text": "1908"}]}
        e.savel(path, [shifted])
        with self.assertRaisesRegex(AssertionError, "Gold span text mismatch"):
            e.load_gold(path, items, old, records)

    def test_fit_thresholds_calls_only_validation_cohort_without_test_gold_files(self):
        f = fixture("ab", [(0, 1), (1, 2)], [(0, 2, 1, "resolved", [(1, 2)])])
        tokens, spans = align_score(f, [.1, .9])
        for t in tokens:
            t["scores"] = {"hidden_probe__token": t["scores"]["x"], "hidden_probe__broadcast": .8}
        called = []
        def only_validation(root, r7, split, module, models):
            called.append(split)
            self.assertEqual(split, "validation")
            return tokens, spans, {"items": 1}, {}
        module = SimpleNamespace(METHODS=["hidden_probe"])
        original = {"thresholds": {"hidden_probe": {"threshold": .6}}}
        with patch.object(e, "load_r7", return_value=(module, original, {})), patch.object(e, "cohort", side_effect=only_validation), \
             patch.object(e, "source_hashes", return_value={}), patch.object(e, "TOKEN_METHODS", ("hidden_probe",)), \
             contextlib.redirect_stdout(io.StringIO()):
            e.fit_thresholds(self.root, self.root/"not_used")
        self.assertEqual(called, ["validation"])
        frozen = json.loads((self.root/"results/freeze8.json").read_text())
        self.assertFalse(frozen["heads_retrained"])
        self.assertEqual(frozen["original_thresholds"]["hidden_probe__token"]["threshold"], .6)
        self.assertEqual(frozen["validation_token_thresholds"]["hidden_probe__token"]["threshold"], .9)
        self.assertFalse((self.root/"data/spans_test.jsonl").exists())
        self.assertFalse((self.root/"data/spans_external_test.jsonl").exists())
        with self.assertRaises(AssertionError):
            e.fit_thresholds(self.root, self.root/"not_used")

    def test_all_span_hashes_lock_without_parsing_heldout_files(self):
        (self.root/"data").mkdir()
        checksums = {}
        for split in ("validation", "test", "external_test"):
            path = self.root/"data"/("spans_"+split+".jsonl")
            # Deliberately not JSON. Hash locking must not parse any held-out labels.
            path.write_bytes(("unparsed canonical bytes for "+split).encode())
            checksums[split] = e.sha(path)
        e.save(self.root/"data/annotation_freeze.json", {"status": "frozen", "canonical_spans_sha256": checksums})
        self.assertEqual(e.annotation_lock(self.root)["canonical_spans_sha256"], checksums)
        (self.root/"data/spans_external_test.jsonl").write_bytes(b"changed labels")
        with self.assertRaisesRegex(AssertionError, "Frozen localization gold changed: external_test"):
            e.annotation_lock(self.root)

    def test_unfrozen_annotations_stop_before_models_or_cohort_scoring(self):
        with patch.object(e, "source_hashes", side_effect=AssertionError("annotations not frozen")), \
             patch.object(e, "load_r7") as model_loader, patch.object(e, "cohort") as scorer:
            with self.assertRaisesRegex(AssertionError, "annotations not frozen"):
                e.fit_thresholds(self.root, self.root/"not_used")
        model_loader.assert_not_called()
        scorer.assert_not_called()


class NativeMappingTests(unittest.TestCase):
    def test_synthetic_redeep_direction_scale_and_raw_token_order_use_original_scorer(self):
        # Import original code only: no real R7 models, features or labels are loaded.
        original = e.import_r7(e.R7_ROOT)
        model = {"kind": "redeep", "heads": np.array([1]), "layers": np.array([0]),
                 "e_min": 1., "e_range": 4., "p_min": 2., "p_range": 2., "beta": .5}
        arrays = {"redeep_ecs": np.array([[9., 1.], [9., 5.]], dtype=np.float32),
                  "redeep_pks": np.array([[2.], [6.]], dtype=np.float32),
                  "lumina_score": np.array([.25, -.5], dtype=np.float32),
                  "mean_nll": np.array([.75, .125], dtype=np.float32),
                  "mean_entropy": np.array([.5, .25], dtype=np.float32)}
        bank = e.TokenBank(arrays)
        items = [{"token_index": 0}, {"token_index": 1}]
        np.testing.assert_array_equal(original.score_method(model, bank, items), [0., 1.5])
        for feature in ("lumina_score", "mean_nll", "mean_entropy"):
            result = original.score_method({"kind": "raw", "feature": feature}, bank, items)
            np.testing.assert_array_equal(result, arrays[feature])


if __name__ == "__main__":
    unittest.main()
