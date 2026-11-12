"""Read-only, fixed-rule case selection; refuses to inspect OOF before completion.

This script produces the selected source/answer/label packet and exact counts.
The accompanying explanation is written after reading these selected cases.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "round16_dataset_expansion/data"
OLD, NEW = "base", "base_all"


def read(path):
    return json.loads(path.read_text("utf-8"))


def readl(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows, method):
    result = dict(tp=0, fp=0, fn=0, tn=0)
    for row in rows:
        y, p = bool(row["gold"]), bool(row["predictions"][method])
        result["tp" if y and p else "fp" if p else "fn" if y else "tn"] += 1
    result["errors"] = result["fp"] + result["fn"]
    result["windows"] = len(rows)
    return result


def merged_ranges(rows, method):
    ranges = []
    for a, b in sorted((r["start"], r["end"]) for r in rows if r["predictions"][method]):
        if ranges and a <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], b)
        else:
            ranges.append([a, b])
    return ranges


def main():
    out = ROOT / "results"
    gate = out / "complete19.json"
    if not gate.exists():
        print("WAITING_FOR_COMPLETE19_NO_PREDICTIONS_READ")
        return
    frozen = read(gate)
    for name in ("window_scores_oof.jsonl", "answer_scores_oof.jsonl"):
        assert sha(out / name) == frozen["files_sha256"][name]
    groups = defaultdict(list)
    for row in readl(out / "window_scores_oof.jsonl"):
        if row["main_eligible"]:
            assert len(row["item_ids"]) == 1
            groups[row["item_ids"][0]].append(row)
    answers = {a["item_id"]: a for a in readl(out / "answer_scores_oof.jsonl")}
    ranking = []
    for iid, rows in groups.items():
        old, new = counts(rows, OLD), counts(rows, NEW)
        ranking.append({"item_id": iid, "error_reduction": old["errors"] - new["errors"],
                        "base": old, "base_all": new})
    better = sorted((r for r in ranking if r["error_reduction"] > 0),
                    key=lambda r: (-r["error_reduction"], r["item_id"]))[:2]
    worse = sorted((r for r in ranking if r["error_reduction"] < 0),
                   key=lambda r: (r["error_reduction"], r["item_id"]))[:2]
    selected_ids = {r["item_id"] for r in better + worse}
    labels_freeze = read(DATA / "annotation_freeze.json")
    dataset_path = DATA / "dataset_train.jsonl"
    assert sha(dataset_path) == labels_freeze["files_sha256"]["data/dataset_train.jsonl"]
    # This source file contains only the actual R16 training split.
    selected = {r["annotation"]["item_id"]: r for r in readl(dataset_path)
                if r["annotation"]["item_id"] in selected_ids}
    assert set(selected) == selected_ids
    result = {"scope": "Only fixed R19 OOF and R16 actual train source/answer/frozen labels",
              "selection_rule": "Per answer: base(FP+FN) minus base_all(FP+FN); highest two strictly positive and lowest two strictly negative; tie item_id ascending; no padding",
              "unit_note": "Overlapping 4 raw-BPE windows are not independent facts",
              "complete19_sha256": sha(gate), "dataset_train_sha256": sha(dataset_path),
              "eligible_answers_ranked": len(ranking), "cases": [],
              "ranking": sorted(ranking, key=lambda r: (-r["error_reduction"], r["item_id"]))}
    for direction, case_rows in (("improved", better), ("regressed", worse)):
        for rank, selected_row in enumerate(case_rows, 1):
            iid = selected_row["item_id"]
            ready = selected[iid]
            inp, ann, response = ready["input"], ready["annotation"], ready["response"]
            assert inp["split"] == ann["split"] == "train"
            source_path = DATA / "generation_records" / (inp["row_id"] + ".json")
            generation = read(source_path)
            assert sha(source_path) == ready["source_generation_sha256"] == ann["source_generation_sha256"]
            assert generation["response"] == response
            assert generation["response_token_ids"] == ready["response_token_ids"]
            assert generation["response_token_offsets"] == ready["response_token_offsets"]
            assert ann["localization_status"] == "resolved"
            for span in ann["risk_spans"]:
                assert response[span["start"]:span["end"]] == span["text"]
            rows = sorted(groups[iid], key=lambda r: (r["start"], r["end"], r["window_key"]))
            item = answers[iid]
            assert item["main_eligible"] and item["gold"] == ann["original_risk"]
            assert all(r["row_id"] == inp["row_id"] for r in rows)
            changes = []
            for row in rows:
                assert response[row["start"]:row["end"]] == row["text"]
                if row["predictions"][OLD] == row["predictions"][NEW]:
                    continue
                change = ("tp_gained" if row["gold"] else "fp_added") if row["predictions"][NEW] else ("tp_lost" if row["gold"] else "fp_removed")
                changes.append({k: row[k] for k in ("window_key", "start", "end", "text", "gold", "scores", "predictions")})
                changes[-1]["change"] = change
            result["cases"].append({**selected_row, "direction": direction, "selection_rank": rank,
                "row_id": inp["row_id"], "question_id": inp["question_id"], "group_id": inp["group_id"],
                "category": inp["category"], "condition": inp["condition"], "fold": item["fold"],
                "visible_input": {k: inp[k] for k in ("system", "prompt", "questions", "passages")},
                "response": response, "answer_text": item["text"], "frozen_annotation": ann,
                "answer_judgment": {"gold": item["gold"], "base": {"score": item["scores"][OLD], "predicted_risk": item["predictions"][OLD]},
                                    "base_all": {"score": item["scores"][NEW], "predicted_risk": item["predictions"][NEW]}},
                "changed_windows": changes,
                "highlight_ranges": {method: [{"start": a, "end": b, "text": response[a:b]}
                                        for a, b in merged_ranges(rows, method)] for method in (OLD, NEW)},
                "interpretation_status": "awaiting_source_answer_label_reading"})
    target = out / "CASE_REVIEW19.json"
    assert not target.exists(), "Do not overwrite the reviewed case document"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": [{k:c[k] for k in ("item_id", "direction", "error_reduction", "base", "base_all")} for c in result["cases"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
