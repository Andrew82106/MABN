"""Synthetic regressions only; never trains on or opens heldout experiment labels."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

PATH = Path(__file__).resolve().parents[1]/"src/evaluate10.py"
spec = importlib.util.spec_from_file_location("round10_eval_tests", PATH)
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)


def tok(n=0, group="g", condition="complete", item="i", gold=0, risky=False):
    return {"token_key": f"{group}_{condition}_t{n}", "token_index": n, "token_id": n,
        "row_id": f"{group}_{condition}", "question_id": group, "group_id": group,
        "split": "train", "condition": condition, "item_ids": [item], "gold": gold,
        "main_eligible": True, "risk_item_eligible": risky, "lexical": True,
        "attribute_group": "time", "scores": {}, "characters": [n], "risk_characters": [n] if gold else [],
        "text": "a", "start": n, "end": n+1, "abstention_overlap": False,
        "unresolved_overlap": False, "outside_items": False}


def item(group="g", condition="complete", iid="i", gold=0, safe=False):
    return {"item_id": iid, "row_id": f"{group}_{condition}", "question_id": group,
        "group_id": group, "condition": condition, "split": "train", "category": "time",
        "gold": gold, "main_eligible": True, "asserted_eligible": not safe,
        "reviewed_safe_refusal": safe, "original_stance": "abstained" if safe else "asserted",
        "scores": {}, "parse_ok": True, "start": 0, "end": 1, "text": "a"}


def populate(rows, names, values):
    for row, value in zip(rows, values):
        row["scores"] = {n: value for n in names}


def thresholds():
    return {n: {"threshold": .5} for n in e.ITEM_METHODS+e.LOCALIZATION_METHODS}


class WeightAndMeanTests(unittest.TestCase):
    def test_long_answer_cannot_dominate_training_group_or_scaler(self):
        rows = [tok(i, group="long", gold=i%2) for i in range(100)]+[tok(group="short")]
        base = e.base_weights(rows, "token")
        self.assertAlmostEqual(base[:100].sum(), base[100:].sum())
        scaler = StandardScaler().fit(np.asarray([10.]*100+[0.])[:, None], sample_weight=base)
        self.assertAlmostEqual(scaler.mean_[0], 5.)
        weights, _ = e.loss_weights(rows, np.asarray([r["gold"] for r in rows]), base)
        self.assertAlmostEqual(weights[:100].sum(), weights[100:].sum())
        self.assertAlmostEqual(weights.mean(), 1.)

    def test_condition_equal_base_mass_not_proportional_to_token_count(self):
        rows = [tok(i) for i in range(12)]+[tok(condition="partial", item="j")]
        w = e.base_weights(rows, "token")
        self.assertAlmostEqual(w[:12].sum(), w[12:].sum())
        items = [item(), item(condition="partial", iid="j")]
        np.testing.assert_allclose(e.base_weights(items, "item"), [1., 1.])

    def test_original_item_mean_includes_punctuation_and_ignores_gold(self):
        bank = e.Bank(Path("unused"), {"r": ({"response_token_offsets": [[0, 1], [1, 2], [2, 3]]}, "sha")})
        bank.cache["r"] = {"lb": np.asarray([[0.], [10.], [2.]], dtype=np.float32)}
        row = {"row_id": "r", "start": 0, "end": 3, "parse_ok": True, "gold": 1}
        # Middle token is punctuation; original Lookback item adapter still includes it.
        self.assertEqual(float(bank.item_matrix([row], "lb")[0, 0]), 4.)
        row["gold"] = None
        self.assertEqual(float(bank.item_matrix([row], "lb")[0, 0]), 4.)
        row["parse_ok"] = False
        self.assertTrue(np.isnan(bank.item_matrix([row], "lb")).all())

    def test_fine_training_does_not_reuse_item_labels_and_broadcast_threshold_fixed(self):
        class FakeBank:
            def item_matrix(self, rows, feature):
                return np.asarray([[r["x"]] for r in rows], dtype=np.float32)
            token_matrix = item_matrix
        items = [item("a", gold=0, safe=True), item("b", gold=1)]
        tokens = [tok(group="b", gold=0, risky=True), tok(1, group="b", gold=1, risky=True)]
        for rr in (items, tokens):
            for row, x in zip(rr, (-1., 1.)):
                row["x"] = x
        models, limits, candidates, audit = e.fit_models(FakeBank(), items, tokens, items, tokens)
        self.assertEqual(len(models), 10); self.assertEqual(len(limits), 15)
        self.assertEqual(set(models), set(e.ITEM_METHODS+e.TOKEN_METHODS))
        self.assertTrue(all(len(c) == 2 for c in candidates.values()))
        self.assertEqual(audit["item"]["observations"], 2)
        self.assertEqual(audit["token"]["observations"], 2)
        for dest, source in e.BROADCAST.items():
            self.assertEqual(limits[dest]["threshold"], limits[source]["threshold"])
            self.assertEqual(limits[dest]["threshold_origin"], source)


class CountTests(unittest.TestCase):
    def test_question_main_includes_safe_refusal_fp_but_asserted_secondary_does_not(self):
        rows = [item("a", gold=1), item("b", gold=0), item("c", gold=0, safe=True)]
        populate(rows, e.ITEM_METHODS, [.9, .1, .9])
        result = e.evaluate(rows, [], [], thresholds())
        main = result["answer_methods"]["lb__item"]
        self.assertEqual((main["micro"]["tp"], main["micro"]["fp"], main["micro"]["tn"]), (1, 1, 1))
        self.assertAlmostEqual(main["micro"]["f1"], 2/3)
        self.assertEqual(main["asserted_only"]["f1"], 1.)
        self.assertEqual(main["reviewed_safe_refusals"]["nonrisk_false_positive_rate"], 1.)

    def test_precise_numeric_bpe_function_words_and_normal_answer_fp(self):
        text = "It was 2020."
        g = {"row_id": "r", "question_id": "q", "response": text,
             "response_token_ids": list(range(5)), "response_token_offsets": [[0, 2], [2, 6], [6, 9], [9, 11], [11, 12]]}
        row = {**item(), "row_id": "r", "question_id": "q", "group_id": "q", "end": len(text), "text": text}
        gold = {"i": {"localization_status": "resolved", "risk_spans": [{"start": 7, "end": 11, "text": "2020"}]}}
        old = {"i": {"annotation": {"risk": 1, "stance": "asserted"}}}
        tokens, regions = e.metric.align_row(g, [row], gold, old)
        self.assertEqual([t["gold"] for t in tokens if t["main_eligible"]], [0, 0, 1, 1])
        populate(tokens, ["x"], [.1, .9, .9, .1, .9])
        normal = tok(group="normal"); normal["scores"] = {"x": .9}; tokens.append(normal)
        m = e.metric.localization_metrics(tokens, regions, "x", .5)
        self.assertEqual([m["micro"][k] for k in ("tp", "fp", "fn", "tn")], [1, 2, 1, 1])
        self.assertEqual(m["span_regions"]["mean_token_coverage"], .5)

    def test_missing_risk_score_remains_fn(self):
        rr = [tok(gold=1), tok(1, gold=1), tok(2, gold=0)]
        populate(rr, ["x"], [.9, None, .1])
        m = e.metric.confusion(rr, "x", .5)
        self.assertEqual(m["missing_predictions"], 1)
        self.assertEqual(m["fn"], 1); self.assertAlmostEqual(m["f1"], 2/3)
        selection = e.threshold_search([1, 1, 0], [.9, np.nan, .1])
        self.assertAlmostEqual(selection["validation_f1"], 2/3)

    def test_broadcast_cannot_gain_within_answer_ranking(self):
        rr = [tok(gold=0, risky=True), tok(1, gold=1, risky=True)]
        populate(rr, ["x"], [.9, .9])
        self.assertEqual(e.within_answer(rr, "x")["mean_auroc"]["mean"], .5)

    def test_bootstrap_keeps_both_conditions_together_at_both_granularities(self):
        ii = [item(gold=1), item(condition="partial", iid="j", gold=1)]
        tt = [tok(gold=1, risky=True), tok(condition="partial", item="j", gold=1, risky=True)]
        populate(ii, e.ITEM_METHODS, [.9, .1]); populate(tt, e.LOCALIZATION_METHODS, [.9, .1])
        b = e.bootstrap(ii, tt, thresholds(), draws=37)
        self.assertEqual(b["groups"], 1)
        for subset, name in (("answer_items", "lb__item"), ("all_resolved_items", "lb__token")):
            np.testing.assert_allclose(b["subsets"][subset]["methods"][name]["f1"]["ci95"], [2/3, 2/3])
            np.testing.assert_allclose(b["subsets"][subset]["methods"][name]["recall"]["ci95"], [.5, .5])
            np.testing.assert_allclose(b["subsets"][subset]["contrasts"]["primary_full_minus_lb"]["f1"]["ci95"], [0., 0.])


class GoldAndBankTests(unittest.TestCase):
    def label_fixture(self, root):
        (root/"data").mkdir()
        it = {**item(), "item_index": 1, "text": "A 20", "end": 4}
        a = {k: it[k] for k in ("item_id", "row_id", "question_id", "split", "text", "start", "end")}
        a.update(source_generation_sha256="current", original_stance="asserted", original_risk=1,
                 localization_status="resolved", risk_spans=[{"start": 2, "end": 4, "text": "20"}])
        e.savel(root/"data/annotations_train.jsonl", [a])
        return it, a, {it["row_id"]: ({"response": "A 20"}, "current")}

    def test_stale_hash_or_shifted_text_cannot_reuse_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); it, a, records = self.label_fixture(root)
            e.load_gold(root, "train", [it], records)
            a["source_generation_sha256"] = "old"; e.savel(root/"data/annotations_train.jsonl", [a])
            with self.assertRaisesRegex(AssertionError, "Stale"):
                e.load_gold(root, "train", [it], records)
            a["source_generation_sha256"] = "current"; a["text"] = "B 20"
            e.savel(root/"data/annotations_train.jsonl", [a])
            with self.assertRaisesRegex(AssertionError, "identity"):
                e.load_gold(root, "train", [it], records)

    def test_load_only_requested_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); it, _, records = self.label_fixture(root)
            (root/"data/annotations_test.jsonl").write_text("INVALID TEST JSON, MUST NOT PARSE", encoding="utf-8")
            e.load_gold(root, "train", [it], records)

    def test_safe_refusal_requires_explicit_split_review_not_just_stance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); it, a, _ = self.label_fixture(root)
            a.update(original_stance="abstained", original_risk=None, localization_status="excluded", risk_spans=[])
            decision = root/"data/safe_refusals_train.json"
            e.save(decision, {"safe_refusal_item_ids": []})
            e.save(root/"data/question_label_policy.json", {"status": "reviewed_frozen", "reviewed_safe_refusal_files_sha256": {"train": e.sha(decision)}})
            (root/"data/safe_refusals_test.json").write_text("INVALID TEST JSON", encoding="utf-8")
            self.assertEqual(e.question_policy(root, "train", {it["item_id"]: a}), set())
            e.save(decision, {"safe_refusal_item_ids": [it["item_id"]]})
            e.save(root/"data/question_label_policy.json", {"status": "reviewed_frozen", "reviewed_safe_refusal_files_sha256": {"train": e.sha(decision)}})
            self.assertEqual(e.question_policy(root, "train", {it["item_id"]: a}), {it["item_id"]})
            a["original_risk"] = 1
            with self.assertRaises(AssertionError):
                e.question_policy(root, "train", {it["item_id"]: a})

    def test_bank_real_npz_paths_and_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); g = {"response_token_ids": [10, 11], "response_token_offsets": [[0, 1], [1, 2]]}
            for folder in ("features", "soft_features"):
                path = root/"data"/folder/"r.npz"; path.parent.mkdir(parents=True)
                arrays = {"token_ids": np.asarray([10, 11]), "token_start": np.asarray([0, 1]), "token_end": np.asarray([1, 2])}
                side = {"source_generation_sha256": "g", "row_id": "r"}
                if folder == "features":
                    arrays.update(lookback_features=np.zeros((2, 784)), binding_features=np.ones((2, 32)))
                    side["binding_feature_names"] = [f"b{i}" for i in range(32)]
                else:
                    arrays.update(new_features=np.full((2, 16), 2.), surface_features=np.full((2, 8), 3.))
                    side["new_feature_names"] = [f"n{i}" for i in range(16)]
                    side["surface_feature_names"] = [f"s{i}" for i in range(8)]
                np.savez(path, **arrays); side["arrays_sha256"] = e.sha(path); e.save(path.with_suffix(".json"), side)
            bank = e.Bank(root, {"r": (g, "g")}); row = bank.row("r")
            self.assertEqual(row["lb_full"].shape, (2, 808))
            np.testing.assert_allclose(row["lb_full"][:, 784:792], 3.)
            np.testing.assert_allclose(row["lb_full"][:, 792:], 2.)
            path = root/"data/soft_features/r.npz"
            arrays["token_end"] = np.asarray([1, 3]); np.savez(path, **arrays)
            side["arrays_sha256"] = e.sha(path); e.save(path.with_suffix(".json"), side)
            with self.assertRaisesRegex(AssertionError, "phase/offset"):
                e.Bank(root, {"r": (g, "g")}).row("r")

    def test_fit_stage_never_calls_test_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"results").mkdir()
            calls = []
            def fake_cohort(root, split, meta):
                calls.append(split)
                if split == "test":
                    raise AssertionError("Heldout labels were opened")
                return [], [], [], {}
            class FakeBank:
                names = {}
                def __init__(self, *args): pass
            with patch.object(e, "source_hashes", return_value={}), patch.object(e, "metadata", return_value=([], [], {})), \
                 patch.object(e, "cohort", side_effect=fake_cohort), patch.object(e, "Bank", FakeBank), \
                 patch.object(e, "fit_models", return_value=({}, {}, {}, {})), patch.object(e, "predict"), \
                 patch.object(e, "evaluate", return_value={}):
                e.fit(root)
            self.assertEqual(calls, ["train", "validation"])
            self.assertTrue((root/"results/freeze10.json").exists())
            with self.assertRaisesRegex(AssertionError, "overwrite"):
                e.fit(root)


class FreezeGraphTests(unittest.TestCase):
    def setup_graph(self, directory):
        root = Path(directory)/"round10_dual_granularity"; old = root.parent/"round9_evidence_binding"
        (root/"data").mkdir(parents=True); (root/"src").mkdir()
        for name in ("run10.py", "feature10.py", "evaluate10.py"):
            (root/"src"/name).write_text("# fixed fixture source\n", encoding="utf-8")
        dependencies = {}
        for key in e.EXTERNAL_SOURCES:
            path = (root/key).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# dependency\n", encoding="utf-8"); dependencies[key] = path
        (old/"data").mkdir(exist_ok=True)
        for name in ("freeze.json", "annotation_freeze.json"):
            e.save(old/"data"/name, {"status": "frozen"})
        protocol = {"schema": e.SCHEMA, "feature_sets": e.FEATURES, "primary_method": "lb_full",
            "lr": {"C": e.LR_C, "seed": e.SEED, "solver": "liblinear", "penalty": "l2", "max_iter": 2000},
            "bootstrap": {"seed": e.BOOTSTRAP_SEED, "draws": e.BOOTSTRAP_DRAWS},
            "features": {"lookback_dimensions": 784, "new_dimensions": 16, "surface_dimensions": 8, "old_binding_dimensions": 32}}
        e.save(root/"protocol.json", protocol)
        for name in ("PLAN.md", "ANNOTATION_GUIDE.md"):
            (root/name).write_text("fixed protocol\n", encoding="utf-8")
        rows = [{"row_id": f"{split}_{condition}", "split": split, "condition": condition,
                 "group_id": split, "question_id": split, "expected_items": 1, "questions": ["Q?"]}
                for split in ("train", "validation", "test") for condition in ("complete", "partial")]
        e.savel(root/"data/inputs.jsonl", rows)
        e.savel(root/"data/dev_inputs.jsonl", rows[:4])
        reuse = {"excludes_round9_test": True, "source_data_freeze_sha256": e.sha(old/"data/freeze.json"),
            "source_annotation_freeze_sha256": e.sha(old/"data/annotation_freeze.json"),
            "dev_inputs_sha256": e.sha(root/"data/dev_inputs.jsonl"), "rows": {}, "annotation_files": {}}
        base_sources = [root/"src/run10.py", root.parent/"round7_evidence_grounding/src/model7.py",
                        root.parent/"round7_evidence_grounding/src/attention7.py",
                        old/"src/binding9.py", old/"src/run9.py"]
        def signature(stage):
            pp = base_sources+[root/"src/feature10.py"] if stage == "soft" else base_sources
            sig = {"stage": stage, "code_sha256": {str(p.resolve()): e.sha(p) for p in pp}}
            return {"stage_signature": sig, "stage_signature_sha256": e.digest(sig)}
        generations = []
        for row in rows:
            rid = row["row_id"]
            gp = root/"data/generation_records"/(rid+".json")
            g = {"row_id": rid, "input_row_sha256": e.digest(row), **signature("generate")}
            e.save(gp, g); generations.append(g)
            reused = row["split"] != "test"
            file_names = [f"data/generation_records/{rid}.json"]
            for folder, stage in (("features", "core"), ("soft_features", "soft")):
                p = root/"data"/folder/(rid+".npz"); p.parent.mkdir(exist_ok=True)
                p.write_bytes(b"fixed-array-fixture")
                e.save(p.with_suffix(".json"), {"source_generation_sha256": e.sha(gp), "arrays_sha256": e.sha(p),
                    "input_row_sha256": e.digest(row), **signature(stage)})
                if folder == "features":
                    file_names.extend(f"data/{folder}/{rid}.{ext}" for ext in ("json", "npz"))
            if reused:
                for name in file_names:
                    dest = old/name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((root/name).read_bytes())
                reuse["rows"][rid] = {"input_row_sha256": e.digest(row), "files_sha256": {n: e.sha(root/n) for n in file_names}}
        e.savel(root/"data/generated.jsonl", generations)
        spans, refusal_hashes = {}, {}
        for split in ("train", "validation", "test"):
            name = f"data/annotations_{split}.jsonl"
            # Deliberately not JSON: source_hashes must treat test decisions as opaque bytes.
            (root/name).write_text("OPAQUE LABEL BYTES "+split, encoding="utf-8")
            spans[split] = e.sha(root/name)
            safe = root/"data"/f"safe_refusals_{split}.json"; safe.write_text("OPAQUE REVIEW BYTES "+split, encoding="utf-8")
            refusal_hashes[split] = e.sha(safe)
            if split != "test":
                (old/name).write_bytes((root/name).read_bytes()); reuse["annotation_files"][name] = spans[split]
        e.save(root/"data/reuse_manifest.json", reuse)
        e.save(root/"data/question_label_policy.json", {"status": "reviewed_frozen", "reviewed_safe_refusal_files_sha256": refusal_hashes})
        e.save(root/"data/annotation_freeze.json", {"status": "frozen", "canonical_spans_sha256": spans,
            "question_label_policy_sha256": e.sha(root/"data/question_label_policy.json")})
        files = ("protocol.json", "PLAN.md", "ANNOTATION_GUIDE.md", "data/inputs.jsonl", "data/reuse_manifest.json",
                 "src/evaluate10.py", "src/run10.py", "src/feature10.py")
        e.save(root/"data/freeze.json", {"status": "frozen", "created_before_test_generation": True,
            "files_sha256": {n: e.sha(root/n) for n in files},
            "external_source_sha256": {k: e.sha(p) for k, p in dependencies.items()}})
        return root, dependencies

    def test_complete_chain_hashes_test_labels_without_parsing_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, dependencies = self.setup_graph(tmp)
            with patch.object(e, "ROOT", root), patch.object(e, "EXTERNAL_SOURCES", dependencies):
                hashes = e.source_hashes(root)
            self.assertIn("data/annotations_test.jsonl", hashes["files"])
            self.assertIn("data/safe_refusals_test.json", hashes["files"])

    def test_old_test_policy_and_external_code_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, dependencies = self.setup_graph(tmp)
            with patch.object(e, "ROOT", root), patch.object(e, "EXTERNAL_SOURCES", dependencies):
                e.source_hashes(root)
                test_review = root/"data/safe_refusals_test.json"
                before = test_review.read_bytes(); test_review.write_bytes(b"changed after freeze")
                with self.assertRaisesRegex(AssertionError, "Safe-refusal decisions changed"):
                    e.source_hashes(root)
                test_review.write_bytes(before)
                next(iter(dependencies.values())).write_text("changed dependency", encoding="utf-8")
                with self.assertRaisesRegex(AssertionError, "External frozen dependency"):
                    e.source_hashes(root)

    def test_same_generation_relabel_or_reuse_edit_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, dependencies = self.setup_graph(tmp)
            with patch.object(e, "ROOT", root), patch.object(e, "EXTERNAL_SOURCES", dependencies):
                p = root/"data/annotations_test.jsonl"; before = p.read_bytes(); p.write_bytes(b"new labels")
                with self.assertRaisesRegex(AssertionError, "Labels changed"):
                    e.source_hashes(root)
                p.write_bytes(before)
                (root/"data/features/train_complete.npz").write_bytes(b"changed development features")
                with self.assertRaisesRegex(AssertionError, "Changed development reuse"):
                    e.source_hashes(root)


if __name__ == "__main__":
    unittest.main()
