"""Synthetic correctness checks only: never opens experimental scores or labels."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

PATH = Path(__file__).resolve().parents[1]/"src/evaluate9.py"
spec = importlib.util.spec_from_file_location("evaluate9_test_module", PATH)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def token(i, group="g0", condition="complete", item="item0", y=0, coarse=0):
    return {"token_key": f"{group}_{condition}_t{i}", "token_index": i, "token_id": i,
            "row_id": f"{group}_{condition}", "question_id": group, "group_id": group,
            "split": "train", "condition": condition, "item_ids": [item],
            "main_eligible": True, "risk_item_eligible": bool(coarse), "gold": y,
            "coarse_gold": coarse, "lexical": True, "scores": {}, "attribute_group": "time",
            "characters": [i], "risk_characters": [i] if y else [], "text": "a", "start": i, "end": i+1,
            "abstention_overlap": False, "unresolved_overlap": False, "outside_items": False}


class WeightTests(unittest.TestCase):
    def test_long_answer_does_not_dominate_group_or_scaler(self):
        rows = [token(i, "long", item="long", y=i%2, coarse=1) for i in range(100)]
        rows += [token(0, "short", item="short", y=0)]
        b = e.base_weights(rows)
        self.assertAlmostEqual(b[:100].sum(), b[100:].sum())
        x = np.asarray([10.]*100+[0.])[:, None]
        scaler = StandardScaler().fit(x, sample_weight=b)
        self.assertAlmostEqual(scaler.mean_[0], 5.)
        y = np.asarray([r["gold"] for r in rows])
        w = e.loss_weights(rows, y, b, e.class_factors(y, b))
        self.assertAlmostEqual(w.mean(), 1.)
        self.assertAlmostEqual(w[:100].sum(), w[100:].sum())

    def test_conditions_and_items_equal_base_mass(self):
        rows = [token(i, condition="complete", item="a") for i in range(20)]
        rows += [token(i, condition="partial", item="b") for i in range(2)]
        b = e.base_weights(rows)
        self.assertAlmostEqual(b[:20].sum(), b[20:].sum())

    def test_broadcast_does_not_filter_by_annotation(self):
        rows = [token(0), token(1)]
        rows[1].update(main_eligible=False, abstention_overlap=True)
        for r, v in zip(rows, (.1, .9)):
            r["scores"] = {"lb_coarse": v, "binding_coarse": v}
        e.apply_broadcast(rows)
        self.assertEqual(rows[0]["scores"]["lb_coarse_broadcast"], .5)
        self.assertEqual(rows[1]["scores"]["lb_coarse_broadcast"], .5)
        rows[1]["scores"]["lb_coarse"] = None
        e.apply_broadcast(rows)
        self.assertIsNone(rows[0]["scores"]["lb_coarse_broadcast"])


class CountingTests(unittest.TestCase):
    def test_numeric_subtokens_and_function_words_remain(self):
        text = "1. It was 2020."
        offsets = [[0, 1], [1, 2], [2, 5], [5, 9], [9, 12], [12, 14], [14, 15]]
        generated = {"row_id": "r", "question_id": "q", "response": text,
                     "response_token_ids": list(range(7)), "response_token_offsets": offsets}
        item = {"item_id": "r__1", "row_id": "r", "group_id": "q", "question_id": "q", "split": "test",
                "condition": "partial", "start": 3, "end": len(text), "text": text[3:]}
        gold = {"r__1": {"localization_status": "resolved", "risk_spans": [{"start": 10, "end": 14, "text": "2020"}]}}
        original = {"r__1": {"annotation": {"stance": "asserted", "risk": 1}}}
        rows, regions = e.metric.align_row(generated, [item], gold, original)
        eligible = [r for r in rows if r["main_eligible"]]
        self.assertEqual([r["gold"] for r in eligible], [0, 0, 1, 1])
        for r in rows:
            r["scores"]["x"] = .9 if r["token_index"] in (3, 4) else .1
        m = e.metric.localization_metrics(rows, regions, "x", .5)
        self.assertEqual([m["micro"][k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(m["micro"]["f1"], .5)
        self.assertEqual(m["span_regions"]["any_hit_recall"], 1.)
        self.assertEqual(m["span_regions"]["mean_token_coverage"], .5)

    def test_threshold_does_not_drop_missing_known_risk(self):
        entry = e.threshold_search([1, 1, 0], [.8, np.nan, .2])
        self.assertEqual(entry["threshold"], .8)
        self.assertAlmostEqual(entry["validation_f1"], 2/3)

    def test_redeep_direction_train_minmax_not_validation_refit(self):
        model = {"heads": [0], "layers": [0], "e_min": 2., "e_range": 2.,
                 "p_min": 1., "p_range": 4., "beta": .5}
        scores = e.redeep_scores(model, np.asarray([[2.], [4.], [6.]]), np.asarray([[1.], [5.], [9.]]))
        np.testing.assert_allclose(scores, [0., .5, 1.])
        order, corr = e.weighted_rank(np.asarray([[0, 1], [1, 0], [2, -1]]), np.asarray([0, 1, 2]), np.ones(3))
        self.assertEqual(order.tolist(), [0, 1])
        np.testing.assert_allclose(corr, [1, -1])

    def test_item_broadcast_cannot_claim_within_answer_discrimination(self):
        rows = [token(0, y=0, coarse=1), token(1, y=1, coarse=1)]
        for r in rows:
            r["scores"]["x"] = .9
        m = e.within_answer_ranking(rows, "x")
        self.assertEqual(m["mixed_answers"], 1)
        self.assertEqual(m["mean_auroc"]["mean"], .5)

    def test_bootstrap_keeps_two_conditions_together(self):
        rows = [token(0, "g", "complete", "a", y=1, coarse=1), token(0, "g", "partial", "b", y=1, coarse=1)]
        for r in rows:
            r["split"] = "test"
            r["scores"] = {name: float(r["condition"] == "complete") for name in e.METHODS}
        thresholds = {name: {"threshold": .5} for name in e.METHODS}
        result = e.evaluate(rows, [], thresholds)
        boot = e.bootstrap(rows, result["methods"], draws=51)
        ci = boot["subsets"]["all_resolved_items"]["methods"]["lb_fine"]["ci95"]
        np.testing.assert_allclose(ci, [2/3, 2/3])
        self.assertEqual(boot["groups"], 1)


class GoldTests(unittest.TestCase):
    def setup_files(self, folder):
        root = Path(folder); (root/"data").mkdir()
        item = {"item_id": "r__1", "row_id": "r", "question_id": "q", "split": "train",
                "text": "A 20", "start": 0, "end": 4, "parse_ok": True}
        generated = {"response": "A 20"}
        annotation = {**item, "source_generation_sha256": "current", "original_stance": "asserted",
                      "original_risk": 1, "localization_status": "resolved", "risk_spans": [{"start": 2, "end": 4, "text": "20"}]}
        path = root/"data/annotations_train.jsonl"
        path.write_text(json.dumps(annotation)+"\n", encoding="utf-8")
        return root, item, generated, annotation, path

    def test_stale_hash_and_shifted_span_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, item, g, a, path = self.setup_files(tmp)
            e.load_gold(root, "train", [item], {"r": (g, "current")})
            a["source_generation_sha256"] = "old"
            path.write_text(json.dumps(a), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "Stale"):
                e.load_gold(root, "train", [item], {"r": (g, "current")})
            a["source_generation_sha256"] = "current"; a["start"] = 1
            path.write_text(json.dumps(a), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "identity"):
                e.load_gold(root, "train", [item], {"r": (g, "current")})

    def test_train_loader_does_not_open_heldout_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, item, g, _, _ = self.setup_files(tmp)
            (root/"data/annotations_test.jsonl").write_text("INVALID JSON MUST NOT PARSE", encoding="utf-8")
            with patch.object(e, "readl", wraps=e.readl) as spy:
                e.load_gold(root, "train", [item], {"r": (g, "current")})
            self.assertEqual([Path(call.args[0]).name for call in spy.call_args_list], ["annotations_train.jsonl"])

    def test_fit_stage_never_requests_test_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def fake_cohort(_, split, __):
                self.assertIn(split, {"train", "validation"})
                return [], [], {"split": split}
            class FakeBank:
                binding_names = ["synthetic"]
                def __init__(self, *args): pass
            with patch.object(e, "source_hashes", return_value={"synthetic": True}), \
                 patch.object(e, "metadata", return_value=([], [], {})), \
                 patch.object(e, "cohort", side_effect=fake_cohort) as spy, \
                 patch.object(e, "Bank", FakeBank), \
                 patch.object(e, "fit_models", return_value=({}, {}, {}, {})), \
                 patch.object(e, "evaluate", return_value={}):
                e.fit(root)
            self.assertEqual([call.args[1] for call in spy.call_args_list], ["train", "validation"])
            self.assertTrue((root/"results/freeze9.json").exists())


class PipelineTests(unittest.TestCase):
    def test_real_bank_path_binds_ids_offsets_hashes_and_feature_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rid = "r"
            generated = {"response_token_ids": [12, 99], "response_token_offsets": [[0, 1], [1, 3]]}
            arrays = {"features": {"hidden_28": np.zeros((2, 3584)), "lookback_features": np.zeros((2, 784)),
                        "binding_features": np.zeros((2, 3)), "token_nll": np.zeros(2), "token_entropy": np.zeros(2),
                        "token_ids": [12, 99], "token_start": [0, 1], "token_end": [1, 3]},
                      "attention": {"token_redeep_ecs": np.zeros((2, 784)), "token_redeep_pks": np.zeros((2, 28))},
                      "lumina": {"token_lumina_score": np.zeros(2)}}
            for folder, values in arrays.items():
                dest = root/"data"/folder; dest.mkdir(parents=True)
                path = dest/(rid+".npz"); np.savez(path, **values)
                side = {"row_id": rid, "source_generation_sha256": "current", "arrays_sha256": e.sha(path)}
                if folder == "features":
                    side["binding_feature_names"] = ["a", "b", "c"]
                path.with_suffix(".json").write_text(json.dumps(side), encoding="utf-8")
            bank = e.Bank(root, {rid: (generated, "current")})
            self.assertEqual(bank.row(rid)["bound"].shape, (2, 787))
            self.assertEqual(bank.binding_names, ["a", "b", "c"])
            wrong_offsets = {**generated, "response_token_offsets": [[0, 2], [2, 3]]}
            with self.assertRaises(AssertionError):
                e.Bank(root, {rid: (wrong_offsets, "current")}).row(rid)
            with self.assertRaisesRegex(AssertionError, "Stale"):
                e.Bank(root, {rid: (generated, "old")}).row(rid)

    def test_all_sixteen_predictors_share_tokens_and_limited_search(self):
        # Synthetic matrices only; tiny dimensions/epochs deliberately not the formal protocol.
        train, valid = [], []
        for rows, prefix in ((train, "train"), (valid, "val")):
            for group in range(3):
                for condition in ("complete", "partial"):
                    for index in range(4):
                        risk = int(condition == "partial")
                        rows.append(token(index, f"{prefix}{group}", condition, f"{prefix}{group}_{condition}",
                                          y=int(risk and index == 3), coarse=risk))
        class FakeBank:
            def matrix(self, rows, feature):
                widths = {"lookback_features": 5, "bound": 7, "hidden_28": 8, "redeep_ecs": 8, "redeep_pks": 8}
                if feature not in widths:
                    return np.asarray([r["token_index"]+float(r["condition"] == "partial") for r in rows])
                return np.asarray([[np.sin((r["token_index"]+1)*(j+1))+.7*(r["condition"] == "partial")
                                    for j in range(widths[feature])] for r in rows], dtype=np.float32)
        config = {**e.MLP_CONFIG, "widths": (4, 2), "max_epochs": 2, "patience": 1, "batch_size": 8}
        e.torch.set_num_threads(1)
        models, thresholds, candidates, audit = e.fit_models(FakeBank(), train, valid, config)
        self.assertEqual(len(models), 14)
        self.assertEqual(set(thresholds), set(e.METHODS))
        self.assertEqual(len(e.METHODS), 16)
        self.assertEqual(sum(len(candidates[n]) for n in e.LR_METHODS), 10)
        self.assertEqual(sum(len(c["seeds"]) for c in candidates["hidden_mlp"]), 6)
        self.assertEqual(len(candidates["redeep"]), 12)
        self.assertIs(models["lb_fine"]["scaler"], models["lb_coarse"]["scaler"])
        self.assertIs(models["binding_fine"]["scaler"], models["binding_coarse"]["scaler"])
        self.assertEqual(audit["train_token_keys"], [t["token_key"] for t in train])
        self.assertEqual(len({models[n]["lr"] for n in e.MLP_NAMES}), 1)
        self.assertTrue(all(set(t["scores"]) == set(e.METHODS) for t in valid))


class OriginalFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ("data/generation_records", "data/features", "data/attention", "data/lumina", "src"):
            (self.root/folder).mkdir(parents=True)
        self.row = {"row_id": "r", "question_id": "q", "group_id": "q", "system": "synthetic",
                    "prompt": "synthetic question and evidence", "questions": ["Synthetic question?"],
                    "passages": [{"title": "Synthetic", "text": "Synthetic text."}]}
        (self.root/"data/inputs.jsonl").write_text(json.dumps(self.row)+"\n", encoding="utf-8")
        (self.root/"data/lumina_random_manifest.json").write_text("{}", encoding="utf-8")
        (self.root/"src/runtime.py").write_text("SYNTHETIC = True\n", encoding="utf-8")
        for name in ("protocol.json", "PLAN.md", "ANNOTATION_GUIDE.md"):
            (self.root/name).write_text("{}", encoding="utf-8")
        freeze = {"schema": "round9-input-freeze-v1", "status": "frozen", "protocol_file": "protocol.json",
                  "protocol_sha256": e.sha(self.root/"protocol.json"), "expected_rows": 1, "expected_groups": 1,
                  "files_sha256": {n: e.sha(self.root/n) for n in ("data/inputs.jsonl", "data/lumina_random_manifest.json", "src/runtime.py")},
                  "locked_source_files": ["src/runtime.py"],
                  "external_source_sha256": {n: e.sha(p) for n, p in e.EXTERNAL_SOURCES.items()}}
        e.save(self.root/"data/freeze.json", freeze)
        visible = {k: self.row[k] for k in ("system", "prompt", "questions", "passages")}
        self.visible_hash = e.hashlib.sha256(json.dumps(visible, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        expected = {"data_freeze_sha256": e.sha(self.root/"data/freeze.json"),
                    "protocol_sha256": freeze["protocol_sha256"], "visible_input_sha256": self.visible_hash}
        e.save(self.root/"data/generation_records/r.json", {"row_id": "r", **expected})
        (self.root/"data/generated.jsonl").write_text("{}\n", encoding="utf-8")
        for folder in ("features", "attention", "lumina"):
            npz = self.root/"data"/folder/"r.npz"
            # Hash-only freeze checks do not parse the synthetic array bytes.
            npz.write_bytes(b"synthetic unparsed array")
            e.save(npz.with_suffix(".json"), {"row_id": "r", **expected,
                "source_generation_sha256": e.sha(self.root/"data/generation_records/r.json"), "arrays_sha256": e.sha(npz)})
        checks = {}
        for split in ("train", "validation", "test"):
            path = self.root/"data"/f"annotations_{split}.jsonl"
            path.write_text("INVALID GOLD JSON: MUST ONLY HASH HERE", encoding="utf-8")
            checks[split] = e.sha(path)
        e.save(self.root/"data/annotation_freeze.json", {"status": "frozen", "canonical_spans_sha256": checks})

    def test_initial_freeze_links_and_all_gold_only_hashed(self):
        with patch.object(e, "readl", wraps=e.readl) as spy:
            signed = e.source_hashes(self.root)
        self.assertEqual(len(signed["external_source_sha256"]), 4)
        self.assertFalse(any("annotations_" in str(c.args[0]) for c in spy.call_args_list))

    def test_changed_inputs_cannot_be_resigned_by_fit(self):
        with (self.root/"data/inputs.jsonl").open("a", encoding="utf-8") as f:
            f.write("\n")
        with self.assertRaisesRegex(AssertionError, "Original data/code freeze changed"):
            e.source_hashes(self.root)

    def test_sidecar_from_other_freeze_rejected(self):
        path = self.root/"data/attention/r.json"
        side = json.loads(path.read_text(encoding="utf-8"))
        side["data_freeze_sha256"] = "another freeze"
        e.save(path, side)
        with self.assertRaisesRegex(AssertionError, "Feature belongs"):
            e.source_hashes(self.root)

    def test_generation_from_other_protocol_rejected(self):
        path = self.root/"data/generation_records/r.json"
        g = json.loads(path.read_text(encoding="utf-8"))
        g["protocol_sha256"] = "another protocol"
        e.save(path, g)
        with self.assertRaisesRegex(AssertionError, "Generation belongs"):
            e.source_hashes(self.root)

    def test_changed_external_dependency_rejected_without_editing_old_rounds(self):
        fake = self.root/"changed_dependency.py"
        fake.write_text("different synthetic code", encoding="utf-8")
        names = dict(e.EXTERNAL_SOURCES)
        names[next(iter(names))] = fake
        with patch.object(e, "EXTERNAL_SOURCES", names):
            with self.assertRaisesRegex(AssertionError, "Round7/8 dependency changed"):
                e.source_hashes(self.root)


if __name__ == "__main__":
    unittest.main()
