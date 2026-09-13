"""Gold-only geometry ceiling for the frozen atomic microclaim partition.

This is not a detector.  It asks how well the deterministic microclaim spans
could localize the unchanged four-BPE labels if an oracle supplied each
microclaim's risk label.  The official test is never addressed.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "src"
ATOMIC_DIR = ROOT / "research/atomic_relation_audit_r32_v1"
sys.path.insert(0, str(SRC))
import run_development as q  # noqa: E402


def load_atomic_module():
    path = ATOMIC_DIR / "audit_atomic_relations.py"
    spec = importlib.util.spec_from_file_location("atomic_relation_audit_v1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(y, pred):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.count_nonzero(y & pred))
    fp = int(np.count_nonzero(~y & pred))
    fn = int(np.count_nonzero(y & ~pred))
    tn = int(np.count_nonzero(~y & ~pred))
    return {
        "n": len(y), "positive": int(y.sum()), "tp": tp, "fp": fp,
        "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def run():
    assert not (HERE / "AUDIT.json").exists(), "Preserve the existing audit"
    atomic = load_atomic_module()
    meta = q.metadata()
    claims = []
    for partition in ("fit", "calibration"):
        claims.extend(read_jsonl(ATOMIC_DIR / f"microclaims_{partition}.jsonl"))
    by_response = defaultdict(list)
    for claim in claims:
        by_response[claim["response_id"]].append(claim)
    assert len(claims) == 11329 and set(by_response) == {a["response_id"] for a in meta["answers"]}

    window_prediction = np.zeros(len(meta["windows"]), dtype=np.int8)
    answer_prediction = np.zeros(len(meta["answers"]), dtype=np.int8)
    claim_statistics = {partition: {"claims": 0, "positive": 0, "mixed": 0, "without_owned_lexical_bpe": 0}
                        for partition in ("fit", "calibration")}

    for answer_index, (answer, token) in enumerate(zip(meta["answers"], meta["tokens"])):
        local = sorted(by_response[answer["response_id"]], key=lambda row: row["microclaim_index"])
        assignment = atomic.assign_tokens_to_claims(token, local)
        assert len(assignment) == token["token_count"]
        risk = np.asarray(token["risk_mask"], dtype=bool)
        lexical = np.asarray(token["lexical_mask"], dtype=bool)
        claim_risk = np.zeros(len(local), dtype=bool)
        mixed = np.zeros(len(local), dtype=bool)
        for claim_id in range(len(local)):
            selected = lexical & (np.asarray(assignment) == claim_id)
            if not selected.any():
                stats = claim_statistics[answer["partition"]]
                stats["without_owned_lexical_bpe"] += 1
                continue
            values = risk[selected]
            claim_risk[claim_id] = values.any()
            mixed[claim_id] = values.any() and not values.all()
        stats = claim_statistics[answer["partition"]]
        stats["claims"] += len(local)
        stats["positive"] += int(claim_risk.sum())
        stats["mixed"] += int(mixed.sum())
        for wi in meta["answer_windows"][answer["response_id"]]:
            ids = {assignment[token_id] for token_id in meta["windows"][wi]["lexical_token_indices"]}
            ids.discard(-1)
            assert ids
            window_prediction[wi] = int(any(claim_risk[claim_id] for claim_id in ids))
        answer_prediction[answer_index] = int(window_prediction[meta["answer_windows"][answer["response_id"]]].max())

    window_y = np.asarray([row["label"] for row in meta["windows"]], dtype=np.int8)
    answer_y = np.asarray([row["label"] for row in meta["answers"]], dtype=np.int8)
    result = {
        "identity": "gold microclaim geometry oracle; not a model result",
        "microclaim_statistics": claim_statistics,
        "partitions": {},
        "source_sha256": {
            "atomic_script": sha(ATOMIC_DIR / "audit_atomic_relations.py"),
            "microclaims_fit": sha(ATOMIC_DIR / "microclaims_fit.jsonl"),
            "microclaims_calibration": sha(ATOMIC_DIR / "microclaims_calibration.jsonl"),
            "gold_manifest": sha(ROOT / "data/gold_manifest.json"),
        },
        "trained_models": 0,
        "thresholds_selected": 0,
        "formal_baselines_modified": False,
        "official_test_opened": False,
    }
    answer_offsets = {"fit": (0, 634), "calibration": (634, 793)}
    for partition in ("fit", "calibration"):
        left, right = meta["bounds"][partition]
        al, ar = answer_offsets[partition]
        result["partitions"][partition] = {
            "windows": counts(window_y[left:right], window_prediction[left:right]),
            "answers": counts(answer_y[al:ar], answer_prediction[al:ar]),
        }
    assert result["partitions"]["fit"]["windows"]["fn"] == 0
    assert result["partitions"]["calibration"]["windows"]["fn"] == 0
    assert result["partitions"]["calibration"]["windows"]["f1"] > .85
    save(HERE / "AUDIT.json", result)
    report = [
        "# 原子微主张定位几何上限", "",
        "这是金标oracle，不是模型成绩：先用原风险词元给每个冻结微主张赋真值，再投影到同一4-BPE窗口。未训练、未选阈值、未读取official test。", "",
        "| 分区 | 窗口F1 | 窗口TP/FP/FN | 整答F1 |", "|---|---:|---:|---:|",
    ]
    for partition in ("fit", "calibration"):
        row = result["partitions"][partition]
        w = row["windows"]
        report.append(f"| {partition} | {w['f1']:.6f} | {w['tp']}/{w['fp']}/{w['fn']} | {row['answers']['f1']:.6f} |")
    report += ["", "calibration窗口上限高于旧句级claim oracle的0.85，说明更细切分能减少错误分数向同句正常文字扩散；真正瓶颈仍是预测微主张是否与资料矛盾。"]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    save(HERE / "complete.json", {
        "status": "complete_verified",
        "files_sha256": {name: sha(HERE / name) for name in ("AUDIT.json", "REPORT.md")},
        "official_test_opened": False,
    })
    print("ATOMIC_RELATION_GEOMETRY_ORACLE_COMPLETE", result["partitions"]["calibration"]["windows"]["f1"])


if __name__ == "__main__":
    run()
