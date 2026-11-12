"""Independent fit-only mapping audit for exact_subset_v4_complement_v1.

Unlike analyze.py, this rebuilds the v4 claim-to-window relation from each
claim's lexical BPE indices and each answer's lexical eligibility. It does not
trust mapped_eligible_window_starts and never opens combined/calibration scores.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXACT = ROOT / "results/exact_subset_attribution_v3_fit_pilot"
V4_DATA = ROOT / "results/atomic_microclaim_relation_expanded_v4"
V4_MODEL = ROOT / "results/microclaim_crossencoder_expanded_v4"
FIT = ROOT / "fit_expansion/data"
N_ANSWERS, N_CLAIMS, N_WINDOWS = 3_680, 34_919, 45_658
THRESHOLD = 0.8225097060203552


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def prefix(path, n):
    output = []
    with Path(path).open("rb") as stream:
        for _ in range(n):
            raw = stream.readline()
            assert raw
            output.append(json.loads(raw.decode("utf-8")))
    return output


def cdf(reference, values):
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="right") / len(ordered)


def main():
    assert not (HERE / "AUDIT.json").exists(), "refuse overwrite"
    result = json.loads((HERE / "RESULTS.json").read_text(encoding="utf-8"))
    with np.load(HERE / "SCORES.npz", allow_pickle=False) as z:
        saved = {name: z[name].copy() for name in z.files}
    keys = list(zip(saved["response_id"].tolist(), saved["token_start"].astype(int).tolist()))
    assert len(keys) == len(set(keys)) == N_WINDOWS
    key_index = {key: i for i, key in enumerate(keys)}

    selected = json.loads((EXACT / "selection_manifest.json").read_text(encoding="utf-8"))["selected"]
    prepared = list(lines(EXACT / "selected_prepared_inputs.jsonl"))
    ids = [row["response_id"] for row in selected]
    assert ids == [row["response_id"] for row in prepared] and len(ids) == 256
    wanted = set(ids)
    tokens = {row["response_id"]: row for row in lines(FIT / "tokens_fit.jsonl")
              if row["response_id"] in wanted}
    windows = [row for row in lines(FIT / "windows_k4_fit.jsonl")
               if row["response_id"] in wanted]
    assert len(windows) == N_WINDOWS and set(tokens) == wanted

    features = np.load(EXACT / "token_features_29.npy", mmap_mode="r", allow_pickle=False)
    slices, cursor = {}, 0
    for row in prepared:
        n = len(row["answer_token_ids"])
        slices[row["response_id"]] = (cursor, cursor + n)
        cursor += n
    assert cursor == len(features) == 46_481
    exact = np.empty(N_WINDOWS, dtype=np.float64)
    gold = np.empty(N_WINDOWS, dtype=np.int8)
    for row in windows:
        key = (row["response_id"], int(row["token_start"]))
        target = key_index[key]
        offset = slices[key[0]][0]
        positions = offset + np.asarray(row["token_indices"], dtype=np.int64)
        exact[target] = -sum(float(features[p, 2]) for p in positions) / 4.0
        gold[target] = int(row["label"])
    assert np.array_equal(gold, saved["label"])

    claim_score = np.full(N_CLAIMS, np.nan, dtype=np.float32)
    claim_fold = np.full(N_CLAIMS, -1, dtype=np.int8)
    for fold in range(5):
        with np.load(V4_MODEL / f"fold_{fold}/predictions.npz", allow_pickle=False) as z:
            indices, values = z["held_indices"], z["held_scores"]
        assert np.isnan(claim_score[indices]).all()
        claim_score[indices] = values
        claim_fold[indices] = fold
    assert np.isfinite(claim_score).all() and np.all(claim_fold >= 0)

    answers = {row["response_id"]: row for row in prefix(V4_DATA / "answers.jsonl", N_ANSWERS)
               if row["response_id"] in wanted}
    examples = defaultdict(list)
    all_examples = prefix(V4_DATA / "examples.jsonl", N_CLAIMS)
    for row in all_examples:
        assert row["partition"] == "fit" and claim_fold[row["example_index"]] == row["held_fold"]
        if row["response_id"] in wanted:
            examples[row["response_id"]].append(row)
    assert set(answers) == set(examples) == wanted

    # Independent projection: derive eligible starts and claim overlap directly.
    v4 = np.full(N_WINDOWS, np.nan, dtype=np.float64)
    edge_count = 0
    for rid in ids:
        answer = answers[rid]
        eligible = [start for start in range(answer["token_count"] - 3)
                    if any(answer["lexical_mask"][start:start + 4])]
        assert set(eligible) == {start for response, start in keys if response == rid}
        for start in eligible:
            owners = []
            for claim in examples[rid]:
                if any(start <= int(token) < start + 4 for token in claim["lexical_bpe_indices"]):
                    owners.append(int(claim["example_index"]))
            assert owners
            edge_count += len(owners)
            v4[key_index[(rid, start)]] = max(float(claim_score[index]) for index in owners)
    assert np.isfinite(v4).all()

    folds = saved["pilot_fold"]
    v4_rank = np.empty(N_WINDOWS, dtype=np.float64)
    exact_rank = np.empty(N_WINDOWS, dtype=np.float64)
    fusion = np.empty(N_WINDOWS, dtype=np.float64)
    exact_pred = np.zeros(N_WINDOWS, dtype=bool)
    fusion_pred = np.zeros(N_WINDOWS, dtype=bool)
    qs = []
    for fold in range(5):
        train, held = folds != fold, folds == fold
        q = float(cdf(v4[train], [THRESHOLD])[0]); qs.append(q)
        v4_rank[held] = cdf(v4[train], v4[held])
        exact_rank[held] = cdf(exact[train], exact[held])
        fusion[held] = .75 * v4_rank[held] + .25 * exact_rank[held]
        exact_pred[held] = exact_rank[held] >= q
        fusion_pred[held] = fusion[held] >= q

    differences = {
        "exact_max_abs": float(np.max(np.abs(exact - saved["exact"]))),
        "v4_max_abs": float(np.max(np.abs(v4 - saved["v4"]))),
        "v4_rank_max_abs": float(np.max(np.abs(v4_rank - saved["v4_rank"]))),
        "exact_rank_max_abs": float(np.max(np.abs(exact_rank - saved["exact_rank"]))),
        "fusion_max_abs": float(np.max(np.abs(fusion - saved["fusion"]))),
    }
    assert max(differences.values()) == 0.0
    assert np.array_equal(saved["v4_pred"], v4 >= THRESHOLD)
    assert np.array_equal(saved["exact_pred"], exact_pred)
    assert np.array_equal(saved["fusion_pred"], fusion_pred)
    assert qs == [row["v4_threshold_training_cdf_q"] for row in result["fold_scaling"]]

    audit = {
        "status": "passed_independent_fit_only_mapping_audit",
        "windows": N_WINDOWS,
        "independently_rebuilt_v4_claim_window_edges": edge_count,
        "method": "Rebuilt eligibility from answer lexical masks and claim coverage from lexical_bpe_indices; did not use mapped_eligible_window_starts.",
        "max_abs_differences": differences,
        "labels_exact": True,
        "predictions_exact": True,
        "fold_cdf_q_exact": True,
        "combined_v4_scores_npz_opened": False,
        "calibration_rows_or_scores_read": False,
        "official_test_read": False,
        "model_loaded": False,
        "GPU_used": False,
        "auditor_sha256": digest(__file__),
        "analyzer_outputs": {
            "scores_sha256": digest(HERE / "SCORES.npz"),
            "results_sha256": digest(HERE / "RESULTS.json"),
            "report_sha256": digest(HERE / "REPORT.md"),
        },
    }
    (HERE / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    complete = json.loads((HERE / "complete.json").read_text(encoding="utf-8"))
    complete["status"] = "complete_fit_only_exploratory_diagnosis_independently_audited"
    complete["audit_sha256"] = digest(HERE / "AUDIT.json")
    (HERE / "complete.json").write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    print("EXACT_V4_COMPLEMENT_INDEPENDENT_AUDIT_PASSED")


if __name__ == "__main__":
    main()
