"""Extend the frozen atomic microclaim transform to all 3,680 fit answers.

The written microclaim file is label blind. Gold-bearing token rows are opened
only after that file is frozen, solely to audit tokenizer ownership coverage.
No calibration/test record or model is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "fit_expansion"
ATOMIC = ROOT / "research/atomic_relation_audit_r32_v1"


def load_atomic():
    path = ATOMIC / "audit_atomic_relations.py"
    spec = importlib.util.spec_from_file_location("atomic_relation_audit_v1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_new(path, rows):
    assert not Path(path).exists(), f"Preserve existing artifact: {path}"
    pending = Path(path).with_suffix(Path(path).suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def save_new(path, value):
    assert not Path(path).exists(), f"Preserve existing artifact: {path}"
    pending = Path(path).with_suffix(Path(path).suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    HERE.mkdir(parents=True, exist_ok=True)
    output = HERE / "microclaims_fit3680.jsonl"
    assert not output.exists()
    atomic = load_atomic()
    raw_path = EXP / "data/fit.jsonl"
    old_plan_path = ROOT / "semantic_baseline/cuda_variant/plans.jsonl"
    new_plan_path = EXP / "minicheck/new_plans.jsonl"
    raw = read_jsonl(raw_path)
    assert len(raw) == 3680 and all(row["partition"] == "fit" for row in raw)
    old_plans = [row for row in read_jsonl(old_plan_path) if row["partition"] == "fit"]
    new_plans = read_jsonl(new_plan_path)
    assert len(old_plans) == 634 and len(new_plans) == 3046
    plans = {row["response_id"]: row for row in old_plans + new_plans}
    assert len(plans) == 3680 and set(plans) == {row["response_id"] for row in raw}

    records = []
    per_answer = []
    reasons = Counter()
    split_parents = 0
    for index, source in enumerate(raw, 1):
        safe = {name: source[name] for name in (
            "source_id", "group_id", "partition", "response_id",
            "original_response", "answer_sha256")}
        made, split_count, reason_count = atomic.make_microclaims(safe, plans[source["response_id"]])
        assert made and all("label" not in row and "risk" not in row for row in made)
        records.extend(made)
        per_answer.append(len(made))
        split_parents += split_count
        reasons.update(reason_count)
        if index % 500 == 0:
            print("EXPANDED_ATOMIC_PREPARE", index, 3680, flush=True)
    write_jsonl_new(output, records)

    # Gold-bearing token rows are deliberately opened only after the label-free
    # artifact is immutable. They are used only to test BPE ownership coverage.
    tokens = read_jsonl(EXP / "data/tokens_fit.jsonl")
    assert len(tokens) == 3680
    by_response = defaultdict(list)
    for row in records:
        by_response[row["response_id"]].append(row)
    owned = lexical = answers_exact = without_owner = 0
    for token in tokens:
        local = sorted(by_response[token["response_id"]], key=lambda row: row["microclaim_index"])
        assignment = atomic.assign_tokens_to_claims(token, local)
        mask = token["lexical_mask"]
        lexical += sum(mask)
        owned += sum(bool(is_lexical and owner >= 0) for is_lexical, owner in zip(mask, assignment))
        without_owner += sum(1 for claim_id in range(len(local)) if claim_id not in assignment)
        answers_exact += int(all((not is_lexical) or owner >= 0 for is_lexical, owner in zip(mask, assignment)))
    assert owned == lexical and answers_exact == 3680

    result = {
        "status": "label_blind_expanded_fit_microclaims_complete",
        "answers": 3680,
        "source_groups": len({row["group_id"] for row in raw}),
        "parent_claims": sum(len(row["claims"]) for row in plans.values()),
        "microclaims": len(records),
        "added_microclaims": len(records) - sum(len(row["claims"]) for row in plans.values()),
        "split_parent_claims": split_parents,
        "split_reasons": dict(sorted(reasons.items())),
        "microclaims_per_answer": {"min": min(per_answer), "max": max(per_answer),
                                     "mean": sum(per_answer) / len(per_answer)},
        "lexical_bpes_owned": owned,
        "lexical_bpes_total": lexical,
        "answers_with_complete_lexical_ownership": answers_exact,
        "microclaims_without_exclusive_bpe_owner": without_owner,
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            raw_path, old_plan_path, new_plan_path, ATOMIC / "audit_atomic_relations.py")},
        "output_sha256": sha(output),
        "labels_used_to_create_microclaims": False,
        "models_or_thresholds": 0,
        "calibration_opened": False,
        "official_test_opened": False,
        "formal_baselines_modified": False,
    }
    save_new(HERE / "preparation.json", result)
    report = [
        "# 扩充fit原子微主张准备", "",
        "同一冻结无标签切分规则已扩展到全部3680份fit回答；calibration和official test均未参与构造。", "",
        f"- 句级主张：{result['parent_claims']}",
        f"- 原子微主张：{result['microclaims']}（新增{result['added_microclaims']}）",
        f"- 材料组：{result['source_groups']}",
        f"- 词面BPE归属：{owned}/{lexical}", "",
        "该文件可作为下一轮微主张证据NLI或关系对训练的fit输入；它没有模型成绩。",
    ]
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    save_new(HERE / "complete.json", {
        "status": "complete",
        "files_sha256": {name: sha(HERE / name) for name in
                           ("microclaims_fit3680.jsonl", "preparation.json", "REPORT.md")},
        "official_test_opened": False,
    })
    print("EXPANDED_ATOMIC_MICROCLAIMS_COMPLETE", len(records), flush=True)


def verify():
    complete = json.loads((HERE / "complete.json").read_text(encoding="utf-8"))
    for name, expected in complete["files_sha256"].items():
        assert sha(HERE / name) == expected, name
    result = json.loads((HERE / "preparation.json").read_text(encoding="utf-8"))
    assert result["answers"] == 3680 and result["lexical_bpes_owned"] == result["lexical_bpes_total"]
    assert not result["calibration_opened"] and not result["official_test_opened"]
    print("EXPANDED_ATOMIC_MICROCLAIMS_VERIFIED")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "verify"))
    args = parser.parse_args()
    {"prepare": prepare, "verify": verify}[args.stage]()
