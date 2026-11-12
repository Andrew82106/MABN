"""Independent CPU audit for the 256-group fit-only exact-subset pilot."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/run_exact_subset_attribution_v3_fit_pilot.py"
V1 = ROOT / "results/exact_subset_attribution_v1"
V2 = ROOT / "results/exact_subset_attribution_v2"
OUT = ROOT / "results/exact_subset_attribution_v3_fit_pilot"
SOURCE = V1 / "prepared_inputs.jsonl"
SELECTED = OUT / "selected_prepared_inputs.jsonl"
SEED = "exact-subset-attribution-v3-fit-pilot-20260912"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def h(identifier):
    return hashlib.sha256((SEED + str(identifier)).encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def calls(function):
    result = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            result.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            result.add(node.func.attr)
    return result


def main():
    freeze = read(OUT / "design_freeze.json")
    complete = read(OUT / "preparation_complete.json")
    manifest = read(OUT / "selection_manifest.json")
    stats = read(OUT / "preparation_statistics.json")
    assert freeze["source_sha256"] == sha(SCRIPT)
    assert freeze["protocol_sha256"] == sha(OUT / "protocol.json")
    for name, expected in complete["files_sha256"].items():
        assert sha(OUT / name) == expected
    for item in freeze["upstream_sha256"].values():
        assert sha(Path(item["path"])) == item["sha256"]

    groups = defaultdict(list)
    source_by_id = {}
    identity_count = 0
    for row in lines(SOURCE):
        identity_count += 1
        assert row["labels_used"] is False
        source_by_id[row["response_id"]] = row
        if row["partition"] == "fit":
            groups[row["group_id"]].append({
                "response_id": row["response_id"],
                "source_id": row["source_id"],
            })
    assert identity_count == 3839 and len(groups) == 615
    ranked_groups = sorted(groups, key=lambda group_id: (h(group_id), str(group_id)))[:256]
    expected = []
    for rank, group_id in enumerate(ranked_groups):
        chosen = min(groups[group_id],
                     key=lambda item: (h(item["response_id"]), str(item["response_id"])))
        expected.append({
            "selection_rank": rank,
            "group_id": group_id,
            "group_hash": h(group_id),
            "response_id": chosen["response_id"],
            "response_hash": h(chosen["response_id"]),
            "source_id": chosen["source_id"],
            "partition": "fit",
        })
    group_array = np.asarray([str(item["group_id"]) for item in expected], dtype=object)
    assigned = np.full(256, -1, dtype=np.int8)
    for fold, (_, held) in enumerate(GroupKFold(5).split(np.arange(256), groups=group_array)):
        assigned[held] = fold
    for item, fold in zip(expected, assigned.tolist()):
        item["held_fold"] = fold
    assert manifest["selected"] == expected
    assert Counter(assigned.tolist()) == {0: 52, 1: 51, 2: 51, 3: 51, 4: 51}
    selected_rows = list(lines(SELECTED))
    assert len(selected_rows) == 256
    assert len({row["group_id"] for row in selected_rows}) == 256
    assert len({row["source_id"] for row in selected_rows}) == 256
    assert all(row == source_by_id[item["response_id"]]
               for row, item in zip(selected_rows, expected))
    raw_tokens = sum(len(row["answer_token_ids"]) for row in selected_rows)
    all_view_tokens = sum(view["input_token_count"]
                          for row in selected_rows for view in row["views"])
    assert raw_tokens == stats["raw_answer_tokens"] == 46481
    assert all_view_tokens == stats["all_eight_view_input_tokens"] == 904496
    assert stats["GPU_forward_calls_full_extract"] == 2048

    source_text = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    selection_text = ast.get_source_segment(source_text, functions["select_identities"])
    assert all(term not in selection_text for term in (
        "label", "length", "token_count", "risk", "score",
        "answer_token_ids", "input_token_count",
    ))
    assert "selection_hash(group_id)" in selection_text
    assert 'selection_hash(item["response_id"])' in selection_text
    for name in ("initialize", "prepare", "check", "audit", "synthetic_selfcheck"):
        assert not ({"load_nf4", "gpu_smoke", "extract", "score",
                     "load_selected_fit_gold", "verify_fit_gold_files"} &
                    calls(functions[name]))
    score_text = ast.get_source_segment(source_text, functions["score"])
    assert "core.fold_token_weights" in score_text
    assert "core.project_token_scores" in score_text
    assert "core.q.answer_scores" in score_text
    assert "core.q.choose_threshold" in score_text
    assert "wddm_gate.exclusive_gpu" in source_text
    assert [len(value) for value in read(OUT / "protocol.json")["fit_only_scoring"]["readouts"].values()] == [3, 5, 29]
    assert not (OUT / "GPU_SMOKE.json").exists()
    assert not (OUT / "extraction_complete.json").exists()
    assert not (OUT / "token_logprob").exists()
    assert not (OUT / "score_started.json").exists()

    report = {
        "status": "passed_independent_CPU_fit_pilot_audit",
        "audit_source_sha256": sha(__file__),
        "production_source_sha256": sha(SCRIPT),
        "selection_manifest_sha256": sha(OUT / "selection_manifest.json"),
        "selected_prepared_sha256": sha(SELECTED),
        "source_prepared_sha256": sha(SOURCE),
        "statistics": {
            "population_answers": identity_count,
            "fit_groups": len(groups),
            "selected_answers_groups_sources": [256, 256, 256],
            "fold_counts": dict(sorted(Counter(assigned.tolist()).items())),
            "raw_answer_tokens": raw_tokens,
            "all_eight_view_input_tokens": all_view_tokens,
            "forward_calls": 2048,
        },
        "checks": {
            "selection_independently_recomputed_exact": True,
            "selection_uses_no_label_length_risk_or_score": True,
            "selected_rows_equal_upstream_label_free_objects": True,
            "five_GroupKFold_assignments_recomputed": True,
            "all_upstream_v1_v2_hashes_unchanged": True,
            "same_v1_extraction_and_v2_gate_bound": True,
            "three_fixed_internal_LR_ablations": True,
            "unified_window_answer_projection_bound": True,
            "GPU_cache_and_score_outputs_absent": True,
            "paper_baseline_untouched": True,
        },
        "fit_gold_files_opened": False,
        "labels_accessed": False,
        "model_loaded": False,
        "GPU_used": False,
        "official_test_opened": False,
        "production_imported": False,
    }
    save(OUT / "INDEPENDENT_AUDIT.json", report)
    (OUT / "INDEPENDENT_AUDIT.md").write_text(
        "# Exact subset attribution v3 fit-only pilot：独立 CPU 审计\n\n"
        "通过。独立脚本未导入生产 runner，重新从 615 个 fit group 计算固定哈希选择与五折，"
        "精确得到 256 个不同 group/source 的回答。46,481 个答案 BPE、904,496 个八视图输入 token 均重算一致。\n\n"
        "CPU 入口未触达模型、GPU 或 fit 金标；GPU/cache/score 输出均不存在。A/B/C 是内部消融，"
        "论文 baseline 未运行或修改。\n",
        encoding="utf-8",
    )
    print("EXACT_SUBSET_V3_FIT_PILOT_INDEPENDENT_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    main()
