"""Read-only claim-geometry ceiling for retrieved-evidence candidate.

Uses development gold only after the label-free claim/evidence plan is frozen.
It trains nothing, selects no hyperparameter, and never addresses official test.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import run_development as q  # noqa: E402


INP = ROOT / "results/retrieved_evidence_nli_v1/inputs.jsonl"
OUT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    prepared = q.lines(INP)
    meta = q.metadata()
    by_response = {row["response_id"]: row for row in prepared}
    assert len(prepared) == 793

    claim_labels: dict[tuple[str, int], int] = {}
    partition_stats = {}
    for partition in ("fit", "calibration"):
        counts = Counter()
        for row in prepared:
            if row["partition"] != partition:
                continue
            risk = np.asarray(meta["by_response"][row["response_id"]]["tokens"]["risk_mask"], bool)
            for claim in row["claims"]:
                values = risk[np.asarray(claim["lexical_token_indices"], int)]
                label = int(values.any())
                claim_labels[(row["response_id"], claim["claim_id"])] = label
                counts["claims"] += 1
                counts["positive_claims"] += label
                counts["partial_positive_claims"] += int(values.any() and not values.all())
                cited = claim["citation_parse"]["status"] == "recognized"
                counts["cited_claims"] += cited
                counts["cited_positive_claims"] += cited and label
                counts["uncited_claims"] += not cited
                counts["uncited_positive_claims"] += (not cited) and label
                zero = all(source["selected"][0]["bm25"] == 0 for source in claim["retrieval"])
                counts["zero_overlap_claims"] += zero
                counts["zero_overlap_positive_claims"] += zero and label
        partition_stats[partition] = dict(counts)

    window_predictions = np.empty(len(meta["windows"]), dtype=np.int8)
    for answer in meta["answers"]:
        row = by_response[answer["response_id"]]
        owners = np.asarray(row["lexical_token_claim"], dtype=int)
        for wi in meta["answer_windows"][answer["response_id"]]:
            claim_ids = {int(owners[ti]) for ti in meta["windows"][wi]["token_indices"] if owners[ti] >= 0}
            assert claim_ids
            window_predictions[wi] = max(claim_labels[(answer["response_id"], ci)] for ci in claim_ids)

    results = {}
    for partition in ("fit", "calibration"):
        lo, hi = meta["bounds"][partition]
        wy = np.asarray([w["label"] for w in meta["windows"][lo:hi]], int)
        wp = window_predictions[lo:hi]
        answer_indices = [i for i, answer in enumerate(meta["answers"]) if answer["partition"] == partition]
        ay = np.asarray([meta["answers"][i]["label"] for i in answer_indices], int)
        ap = np.asarray([
            int(window_predictions[meta["answer_windows"][meta["answers"][i]["response_id"]]].max())
            for i in answer_indices
        ], int)
        results[partition] = {
            "windows": {
                "f1": float(f1_score(wy, wp)),
                "tp": int(((wy == 1) & (wp == 1)).sum()),
                "fp": int(((wy == 0) & (wp == 1)).sum()),
                "fn": int(((wy == 1) & (wp == 0)).sum()),
                "tn": int(((wy == 0) & (wp == 0)).sum()),
            },
            "answers": {
                "f1": float(f1_score(ay, ap)),
                "tp": int(((ay == 1) & (ap == 1)).sum()),
                "fp": int(((ay == 0) & (ap == 1)).sum()),
                "fn": int(((ay == 1) & (ap == 0)).sum()),
                "tn": int(((ay == 0) & (ap == 0)).sum()),
            },
        }

    report = {
        "status": "complete_read_only_development_oracle",
        "definition": "Each automatic claim receives its true any-risk lexical-BPE label, then broadcasts that label through the frozen claim-to-4-BPE mapping.",
        "interpretation": "This is an oracle ceiling for the frozen claim geometry, not a detector result.",
        "partition_claim_counts": partition_stats,
        "oracle_metrics": results,
        "trained_models": 0,
        "selected_hyperparameters": 0,
        "official_test_opened": False,
        "inputs_sha256": sha(INP),
        "source_sha256": sha(Path(__file__)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESULT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cal = results["calibration"]
    text = [
        "# Retrieved-evidence claim geometry ceiling", "",
        "这是冻结自动 claim 切分与统一4-BPE投影的金标上限，不是模型成绩。未训练、未调参、未读取官方test。", "",
        f"- cal窗口oracle F1：**{cal['windows']['f1']:.6f}**",
        f"- cal整答oracle F1：**{cal['answers']['f1']:.6f}**",
        f"- cal窗口 TP/FP/FN：{cal['windows']['tp']}/{cal['windows']['fp']}/{cal['windows']['fn']}",
        "",
        "若窗口上限明显高于0.75，当前瓶颈在claim风险判别信号/读出，而不是4-BPE投影本身。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
