"""Bounded independent CPU replay/count audit of 18 saved citation LR models."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import pickle
import re
import sys
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
QA = OUT.parents[1]
FEATURES = QA / "results/citation_alignment_v1"
FUSION = QA / "results/completed_score_fusion_v1"
sys.path.insert(0, str(FUSION))
from audit_saved_scores import metric, best_threshold


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_features(windows, features):
    """Five fixed fit cases; citation strings/IDs independently specified here."""
    specifications = {("16023", 0): [], ("16023", 2): [("passage 1", [1])],
                      ("11937", 7): [("passages 2-6", [2, 3, 4, 5, 6])],
                      ("14643", 5): [("passage 10", [10])], ("12129", 3): []}
    claims = {(r["response_id"], r["claim_index"]): r for r in rows(FEATURES / "claims.jsonl")}
    text = {r["response_id"]: r for r in rows(QA / "data/fit.jsonl") if r["response_id"] in {k[0] for k in specifications}}
    tokens = {r["response_id"]: r for r in rows(QA / "data/tokens_fit.jsonl") if r["response_id"] in text}
    output = []
    word = re.compile(r"[^\W_]+", re.UNICODE)
    for key, specified in specifications.items():
        claim, row = claims[key], text[key[0]]
        assert claim["partition"] == "fit"
        s = claim["text"]
        references = []
        keep = list(s)
        for literal, ids in specified:
            start = s.index(literal)
            end = start + len(literal)
            references.append((start, end, ids))
            keep[start:end] = " " * (end-start)
        assert [r["ids"] for r in claim["parser"]["references"]] == [r[2] for r in references]
        passages = re.split(r"(?m)^passage ([123]):", row["retrieved_passages"])
        assert passages[0] == "" and passages[1::2] == ["1", "2", "3"]
        source_sets = {int(passages[i]): set(word.findall(passages[i+1].lower())) for i in (1, 3, 5)}
        words = set(word.findall("".join(keep).lower()))
        cover = {i: len(words & source)/len(words) if words else 0. for i, source in source_sets.items()}
        ids = set(i for _, _, numbers in references for i in numbers)
        a = max(cover.values())
        c = max((cover[i] for i in ids if i in cover), default=0.)
        invalid = len(ids-set(source_sets))/len(ids) if ids else 0.
        value = np.array([a, c, a-c if ids else 0., invalid, float(bool(ids))])
        assert np.array_equal(value, claim["features"])
        response = row["original_response"]
        offsets = tokens[key[0]]["response_token_offsets"]
        eligible = []
        reference_chars = {j+claim["start"] for left, right, _ in references for j in range(left, right) if s[j].isalnum()}
        for index, window in enumerate(windows[:168123]):
            if window["response_id"] != key[0]:
                continue
            chars = {j for token_index in window["token_indices"] for j in range(*offsets[token_index]) if response[j].isalnum()}
            if chars and all(claim["start"] <= j < claim["end"] for j in chars):
                overlap = len(chars & reference_chars)/len(chars)
                eligible.append((index, overlap))
        assert eligible
        candidates = [eligible[0], max(eligible, key=lambda t: t[1]), eligible[-1]]
        checked = []
        for index, overlap in dict(candidates).items():
            expected = np.array([*value, overlap, value[2]*overlap, value[3]*overlap], dtype=np.float32)
            assert np.array_equal(expected, features[index]), (key, index)
            checked.append(windows[index]["window_id"])
        output.append({"response_id": key[0], "claim_index": key[1], "specified_ids": sorted(ids),
                       "claim_features_exact": True, "sample_window_features_exact": True,
                       "window_ids": checked, "values": value.tolist()})
    return output


def run():
    complete, summary, started = read(OUT / "complete.json"), read(OUT / "summary.json"), read(OUT / "started.json")
    assert sha(OUT / "summary.json") == complete["summary_sha256"] and summary["fits_completed"] == 18
    assert sha(QA / "src/train_citation_alignment.py") == started["code_sha256"]
    assert sha(OUT / "protocol.json") == started["protocol_sha256"]
    feature_manifest = read(FEATURES / "preparation_complete.json")
    assert sha(FEATURES / "preparation_complete.json") == started["feature_preparation_sha256"]
    assert read(FEATURES / "complete.json") == feature_manifest
    for name, value in feature_manifest["files_sha256"].items():
        assert sha(FEATURES / name) == value
    for path, value in feature_manifest["source_sha256"].items():
        assert sha(path) == value
    feature_freeze = read(FEATURES / "design_freeze.json")
    assert sha(QA / "src/build_citation_alignment.py") == feature_freeze["source_sha256"]
    assert sha(FEATURES / "protocol.json") == feature_freeze["protocol_sha256"]
    assert sha(FUSION / "complete.json") == started["base_scores_complete_sha256"]
    upstream = read(FUSION / "summary.json")
    assert sha(FUSION / "summary.json") == read(FUSION / "complete.json")["summary_sha256"]
    windows = rows(QA / "data/windows_k4_fit.jsonl") + rows(QA / "data/windows_k4_calibration.jsonl")
    answers = rows(QA / "data/answers_fit.jsonl") + rows(QA / "data/answers_calibration.jsonl")
    assert len(windows) == 210364 and len(answers) == 793
    wy = np.array([r["label"] for r in windows]); ay = np.array([r["label"] for r in answers])
    answer_index = {a["response_id"]: i for i, a in enumerate(answers)}
    owner = np.array([answer_index[w["response_id"]] for w in windows])
    assert all(w["partition"] == "fit" for w in windows[:168123])
    assert all(w["partition"] == "calibration" for w in windows[168123:])
    assert len(set(a["group_id"] for a in answers[:634]) & set(a["group_id"] for a in answers[634:])) == 0
    with np.load(FEATURES / "features.npz", allow_pickle=False) as z:
        new = z["features"].copy()
        assert np.array_equal(z["window_ids"], [r["window_id"] for r in windows])
        assert np.array_equal(z["response_ids"], [r["response_id"] for r in windows])
    assert np.array_equal(new, np.load(FEATURES / "window_features.npy")) and new.shape == (210364, 8)
    assert digest([r["window_id"] for r in windows]) == feature_manifest["window_order_sha256"]

    # Independently reconstruct fit-only group/answer/window base mass used by scalers.
    n_windows = Counter(w["answer_id"] for w in windows[:168123])
    n_answers = Counter(a["group_id"] for a in answers[:634])
    base = np.array([1/(n_answers[w["group_id"]]*n_windows[w["answer_id"]]) for w in windows[:168123]])
    base /= base.mean()
    checks, selected = [], {}
    for peer in ("lookback", "harp_claim", "semantic_claim"):
        cols = []
        for alpha in (0., 1.):
            e = next(e for e in upstream["all_candidates"][peer] if e["tail_weight"] == alpha)
            path = FUSION / (e["candidate"] + "_scores.npz")
            assert sha(path) == e["scores_sha256"]
            with np.load(path, allow_pickle=False) as z:
                cols.append(z["window_scores"].copy())
        two = np.column_stack(cols)
        for mode in ("two_scores_only", "two_scores_and_citation"):
            family = peer + "__" + mode
            x = two if mode == "two_scores_only" else np.column_stack((two, new))
            scaler = pickle.loads((OUT / (family + "_scaler.pkl")).read_bytes())
            mean = np.average(x[:168123], weights=base, axis=0)
            var = np.average((x[:168123]-mean)**2, weights=base, axis=0)
            assert np.allclose(mean, scaler.mean_, rtol=1e-12, atol=1e-14)
            assert np.allclose(var, scaler.var_, rtol=1e-12, atol=1e-14)
            assert np.allclose(np.where(var == 0, 1., np.sqrt(var)), scaler.scale_, rtol=1e-12, atol=1e-14)
            scaled = (x-scaler.mean_)/scaler.scale_
            assert np.array_equal(scaled, scaler.transform(x))
            entries = summary["all_candidates"][family]
            assert [e["C"] for e in entries] == [.001, .01, .1]
            keys = []
            for e in entries:
                candidate = e["candidate"]
                assert read(OUT / (candidate + ".json")) == e
                for suffix, hash_key in ((".pkl", "model_sha256"), ("_scores.npz", "scores_sha256")):
                    assert sha(OUT / (candidate + suffix)) == e[hash_key]
                assert sha(OUT / (family + "_scaler.pkl")) == e["scaler_sha256"]
                model = pickle.loads((OUT / (candidate + ".pkl")).read_bytes())
                assert model.n_features_in_ == x.shape[1] == e["input_width"]
                assert model.C == e["C"] and model.random_state == 20261010
                assert model.solver == "liblinear" and model.penalty == "l2" and model.max_iter == 2000
                assert model.classes_.tolist() == [0, 1] and model.n_iter_.tolist() == e["iterations"]
                with np.load(OUT / (candidate + "_scores.npz"), allow_pickle=False) as z:
                    score, answer_score = z["window_scores"].copy(), z["answer_scores"].copy()
                predicted = model.predict_proba(scaled)[:, 1]
                assert np.array_equal(predicted, score)
                manual = expit((scaled @ model.coef_.T + model.intercept_)[:, 0])
                assert np.allclose(manual, score, rtol=0, atol=2e-15)
                maxima = np.full(793, -np.inf); np.maximum.at(maxima, owner, score)
                assert np.array_equal(maxima, answer_score)
                counts = {}
                for partition, wl, wr, al, ar in (("fit", 0, 168123, 0, 634), ("calibration", 168123, 210364, 634, 793)):
                    counts[partition] = {}
                    for unit, y, s, th in (("windows", wy[wl:wr], score[wl:wr], e["thresholds"]["window"]["threshold"]),
                                           ("answers", ay[al:ar], answer_score[al:ar], e["thresholds"]["answer"]["threshold"])):
                        m = metric(y, s, th)
                        assert all(value == e["metrics"][partition][unit][key] for key, value in m.items())
                        counts[partition][unit] = m
                ts = {"window": best_threshold(wy[168123:], score[168123:]), "answer": best_threshold(ay[634:], answer_score[634:])}
                assert ts == e["thresholds"]
                key = [min(ts["window"]["f1"], ts["answer"]["f1"]), ts["window"]["f1"], ts["window"]["precision"], -e["C"]]
                assert key == e["selection_key"]; keys.append(key)
                checks.append({"candidate": candidate, "saved_model_probability_exact": True,
                               "independent_matrix_probability_max_abs_diff": float(np.max(abs(manual-score))),
                               "fit_only_scaler_mean_max_abs_diff": float(np.max(abs(mean-scaler.mean_))),
                               "fit_only_scaler_var_max_abs_diff": float(np.max(abs(var-scaler.var_))),
                               "answer_maxima_exact": True, "thresholds_independently_verified": True, "counts": counts})
            winner = entries[max(range(3), key=lambda i: keys[i])]
            assert winner == summary["selected"][family]
            selected[family] = {"C": winner["C"], "window_f1": winner["metrics"]["calibration"]["windows"]["f1"],
                                "answer_f1": winner["metrics"]["calibration"]["answers"]["f1"]}
    assert len(checks) == 18
    samples = sample_features(windows, new)
    deltas = {peer: {unit: selected[peer+"__two_scores_and_citation"][unit] - selected[peer+"__two_scores_only"][unit]
                    for unit in ("window_f1", "answer_f1")} for peer in ("lookback", "harp_claim", "semantic_claim")}
    result = {"status": "passed", "models_replayed": 18, "windows_per_model": 210364, "answers_per_model": 793,
              "fit_windows": 168123, "fit_answers": 634, "calibration_windows": 42241, "calibration_answers": 159,
              "source_and_feature_hashes_match_completion": True, "feature_row_identity_exact": True,
              "fit_calibration_group_overlap": 0, "checks": checks, "selected": selected, "feature_mode_minus_control": deltas,
              "five_fit_claim_feature_samples": samples,
              "source_code_review": {"parser_and_coverage_inputs": "Answer text and visible source word sets/parsed local citation numbers only",
                                     "gold_type_or_human_span_meta_in_features": False,
                                     "dataset_raw_source_id_in_features": False,
                                     "raw_ids_note": "Local displayed citation IDs are used to resolve a cited passage, then reduced to coverage/invalid fractions; original source IDs and generator identity are not numeric features.",
                                     "training_uses_fit_labels_only": True, "calibration_only_selects_thresholds_and_C": True,
                                     "source_sha256": {p: sha(QA / "src" / p) for p in ("build_citation_alignment.py", "train_citation_alignment.py")}},
              "limitations": ["First feature remains ordinary any-source lexical coverage on no-citation windows. This is not a citation-only single-factor ablation.",
                              "Lexical coverage is not entailment. Upstream fit scores are in-sample, not cross-fitted stacking.",
                              "Only five fixed fit claim/feature cases were independently recomputed; full grammar was not reparsed.",
                              "Partial subsection syntax such as Passage 1:2 is parsed as the top-level passage1; the subsection suffix is not separately flagged unknown. Do not claim exhaustive citation syntax coverage.",
                              "No new fitting, GPU or sealed-test access. AUROC/AP not recomputed in this bounded count audit.",
                              "Repeatedly-used calibration is not independent test evidence; retain all negative comparisons."],
              "new_training": False, "GPU_used": False, "official_test_opened": False,
              "summary_sha256": sha(OUT / "summary.json"), "script_sha256": sha(__file__),
              "independent_metric_utility_sha256": sha(FUSION / "audit_saved_scores.py")}
    (OUT / "INDEPENDENT_REPLAY_CHECK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "models": 18, "selected": selected, "delta": deltas}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
