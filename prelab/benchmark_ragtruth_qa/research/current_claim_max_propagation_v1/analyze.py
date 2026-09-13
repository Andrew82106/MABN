"""One fixed read-only check of propagating the incumbent maximum within claims."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import run_development as q  # noqa: E402


SOURCE = ROOT / "results/large_fixed_convex_v1/semantic_claim__old_tree__large_weight0.4_scores.npz"
PLANS = ROOT / "results/retrieved_evidence_nli_v1/inputs.jsonl"
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    meta = q.metadata()
    rows = {r["response_id"]: r for r in q.lines(PLANS)}
    with np.load(SOURCE, allow_pickle=False) as data:
        source = data["window_scores"].copy()
        old_answers = data["answer_scores"].copy()
    output = np.empty(len(source), dtype=np.float64)
    for answer in meta["answers"]:
        rid = answer["response_id"]
        owners = np.asarray(rows[rid]["lexical_token_claim"], dtype=int)
        window_indices = meta["answer_windows"][rid]
        claim_scores = np.full(len(rows[rid]["claims"]), -np.inf)
        for wi in window_indices:
            claim_ids = {int(owners[t]) for t in meta["windows"][wi]["token_indices"] if owners[t] >= 0}
            for claim_id in claim_ids:
                claim_scores[claim_id] = max(claim_scores[claim_id], source[wi])
        assert np.isfinite(claim_scores).all()
        for wi in window_indices:
            claim_ids = {int(owners[t]) for t in meta["windows"][wi]["token_indices"] if owners[t] >= 0}
            output[wi] = max(claim_scores[claim_id] for claim_id in claim_ids)
    answer_scores = q.answer_scores(meta, output)
    assert np.array_equal(answer_scores, old_answers)
    left, right = meta["bounds"]["calibration"]
    fit_left, fit_right = meta["bounds"]["fit"]
    thresholds = {
        "window": q.choose_threshold([w["label"] for w in meta["windows"][left:right]], output[left:right]),
        "answer": q.choose_threshold(
            [a["label"] for a in meta["answers"] if a["partition"] == "calibration"],
            answer_scores[[i for i, a in enumerate(meta["answers"]) if a["partition"] == "calibration"]],
        ),
    }
    metrics = q.metrics(meta, output, thresholds)
    old_thresholds = {
        "window": q.choose_threshold([w["label"] for w in meta["windows"][left:right]], source[left:right]),
        "answer": thresholds["answer"],
    }
    old_metrics = q.metrics(meta, source, old_thresholds)
    result = {
        "status": "complete_read_only_negative_diagnostic",
        "method": "For each automatic claim, take the maximum incumbent score of every overlapping project window, then assign that maximum to every window overlapping the claim.",
        "selection": "One fixed propagation formula; no alpha/model/feature search. Calibration F1-opt threshold follows the existing common development reporting convention.",
        "incumbent_calibration": old_metrics["calibration"],
        "claim_max_propagation_calibration": metrics["calibration"],
        "answer_scores_exactly_preserved": True,
        "fit_bounds": [fit_left, fit_right],
        "calibration_bounds": [left, right],
        "trained_models": 0,
        "official_test_opened": False,
        "source_sha256": sha(SOURCE),
        "plans_sha256": sha(PLANS),
        "code_sha256": sha(Path(__file__)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    before = old_metrics["calibration"]["windows"]["f1"]
    after = metrics["calibration"]["windows"]["f1"]
    (OUT / "REPORT.md").write_text(
        "# 当前候选 claim-max 传播诊断\n\n"
        f"固定地把每个claim内的最高风险传播到整条claim，cal窗口F1由 **{before:.6f}** 降到 **{after:.6f}**；"
        "整答分数逐值不变。说明单纯扩大报警范围会新增过多误报，不能代替证据关系判别。\n\n"
        "这是反复开发cal上的只读负结果；未训练、未调传播强度、未读取官方test。\n",
        encoding="utf-8",
    )
    print(json.dumps({"before": before, "after": after, "answer_f1": metrics["calibration"]["answers"]["f1"]}))


if __name__ == "__main__":
    main()
