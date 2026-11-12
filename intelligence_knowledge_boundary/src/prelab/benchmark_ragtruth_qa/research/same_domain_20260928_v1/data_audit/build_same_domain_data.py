"""CPU-only, label-blind grouped repartition of already-exposed RAGognize train.

Split assignment is written before labeled conversions are opened. Test targets
are copied into a physically separate sealed file without label statistics.
The predictor input and geometry lines are reused verbatim from audited exports.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import unicodedata

HERE = Path(__file__).resolve().parent
PROJECT = next(p for p in HERE.parents if p.name == "intelligence_knowledge_boundary")
BENCH = PROJECT / "src/prelab/benchmark_ragtruth_qa"
LEGACY = BENCH / "auxiliary_ragognize_train_v1"
CONVERTED = LEGACY / "conversion_v1/all_train"
FIXTURE = PROJECT / "reports/paper/tmp/submission_20260927_v2/external_eval/train_fixture"
OUT = HERE.parent / "data"
SEED = 20260928
VERSION = "ragognize-same-domain-grouped-v1"
INPUT_FIELDS = {
    "schema_version", "response_id", "question", "documents", "retrieved_passages",
    "released_prompt", "original_response", "prompt_prefix", "document_char_spans",
}
FORBIDDEN = {
    "gold", "golden_answer", "labels", "label", "hallucinations", "answerable",
    "analysis", "risk_mask", "answer_risk", "cluelessness", "tags",
    "unanswerability_hint", "unanswerability_hint_setting", "released_hallucinations",
    "risk", "risk_token_count", "window_labels", "answer_label",
}


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def lines(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line), line


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def emit(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def assert_safe(value):
    if isinstance(value, dict):
        assert not (FORBIDDEN & set(value)), "Target field in predictor-visible record"
        for key, item in value.items():
            assert not key.endswith("_analysis")
            assert_safe(item)
    elif isinstance(value, list):
        for item in value:
            assert_safe(item)


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        self.parent[max(a, b)] = min(a, b)


def main():
    if (OUT / "MANIFEST.json").exists():
        raise RuntimeError("Frozen manifest already exists; do not overwrite a data protocol")
    OUT.mkdir(parents=True, exist_ok=True)
    original = [x for x, _ in lines(LEGACY / "GROUP_INDEX.jsonl")]
    metadata = load(FIXTURE / "audit/v2_source_reference_metadata.json")
    inputs = [x for x, _ in lines(FIXTURE / "inputs/answers.jsonl")]
    assert len(original) == len(metadata) == len(inputs) == 1842
    assert [x["train_row_index"] for x in original] == list(range(1842))
    assert [x["row_index"] for x in metadata] == list(range(1842))
    assert [x["response_id"] for x in inputs] == [x["response_id"] for x in metadata]

    dsu, seen = DSU(len(original)), {}
    for i, row in enumerate(original):
        group = row["recommended_group_id"]
        if group in seen:
            dsu.union(i, seen[group])
        else:
            seen[group] = i
    assert len(seen) == 917
    documents, all_document_keys = defaultdict(set), []
    for i, row in enumerate(inputs):
        assert set(row) == INPUT_FIELDS
        assert_safe(row)
        keys = {digest(norm(doc["text"])) for doc in row["documents"]}
        all_document_keys.append(keys)
        for key in keys:
            documents[key].add(i)
    merge_edges = set()
    for i, row in enumerate(metadata):
        for key in row["target_document_sha256"]:
            assert key in all_document_keys[i]
            for j in documents[key]:
                if original[i]["recommended_group_id"] != original[j]["recommended_group_id"]:
                    merge_edges.add(tuple(sorted((i, j))))
                dsu.union(i, j)
    components = defaultdict(list)
    for i in range(len(original)):
        components[dsu.find(i)].append(i)
    group_by_row = {
        i: "same_domain_source_group_" + digest(members)[:16]
        for members in components.values() for i in members
    }
    ordered_groups = sorted(set(group_by_row.values()), key=lambda key: (digest(f"{SEED}|{key}"), key))
    n_holdout = math.ceil(0.15 * len(ordered_groups))
    split_by_group = {
        group: "test" if i < n_holdout else "validation" if i < 2 * n_holdout else "fit"
        for i, group in enumerate(ordered_groups)
    }
    assignments = []
    for i, row in enumerate(original):
        assignments.append({
            "response_id": inputs[i]["response_id"], "train_row_index": i,
            "official_split": "train", "group_id": group_by_row[i],
            "legacy_group_id": row["recommended_group_id"],
            "split": split_by_group[group_by_row[i]],
            "no_hint": not row["unanswerability_hint_setting"],
            "unanswerability_hint_setting": row["unanswerability_hint_setting"],
            "official_user_prompt_index": row["official_user_prompt_index"],
            **{k: metadata[i][k] for k in (
                "normalized_question_sha256", "source_identity_sha256", "source_url_sha256",
                "source_title_sha256", "normalized_context_sha256", "retrieved_context_sha256")},
        })
    by_id = {row["response_id"]: row for row in assignments}
    audit = {"cross_split_identity_hashes": {}, "cross_split_target_document_edges": 0}
    for key in ("normalized_question_sha256", "source_identity_sha256", "source_url_sha256",
                "source_title_sha256", "normalized_context_sha256", "retrieved_context_sha256"):
        occupied = defaultdict(set)
        for row in assignments:
            if row[key] is not None:
                occupied[row[key]].add(row["split"])
        collisions = sum(len(splits) > 1 for splits in occupied.values())
        audit["cross_split_identity_hashes"][key] = collisions
        assert collisions == 0, key
    for i, row in enumerate(metadata):
        for key in row["target_document_sha256"]:
            assert len({assignments[j]["split"] for j in documents[key]} | {assignments[i]["split"]}) == 1
    counts = {}
    for split in ("fit", "validation", "test"):
        selected = [r for r in assignments if r["split"] == split]
        counts[split] = {"answers": len(selected), "groups": len({r["group_id"] for r in selected}),
                         "no_hint_answers": sum(r["no_hint"] for r in selected)}
    with (OUT / "split_assignments.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in assignments:
            emit(handle, row)
    groups = [{"group_id": group_by_row[members[0]], "row_indices": members,
               "legacy_group_ids": sorted({original[i]["recommended_group_id"] for i in members}),
               "split": assignments[members[0]]["split"]} for members in components.values()]
    save(OUT / "GROUPS.json", sorted(groups, key=lambda row: row["group_id"]))
    audit.update({"original_groups": 917, "final_groups": len(components),
                  "largest_group_answers": max(map(len, components.values())),
                  "target_document_merge_edges_across_original_groups": len(merge_edges),
                  "shared_non_target_distractors_do_not_trigger_merges": True,
                  "test_label_statistics_computed": False})
    save(OUT / "GROUP_AUDIT.json", audit)
    save(OUT / "SPLIT_FREEZE_BEFORE_LABEL_COPY.json", {
        "schema_version": VERSION, "seed": SEED, "counts": counts,
        "split_assignment_sha256": file_sha(OUT / "split_assignments.jsonl"),
        "algorithm": "SHA256(str(seed)+'|'+group_id) ascending; first ceil(.15*G) test; next same validation; remainder fit",
        "group_definition": "Transitive closure of prior 917 source/question/context groups plus exact normalized target-document-to-any-document edges over all 1842 rows",
        "label_blind": True, "new_target_labels_opened_before_freeze": False,
        "historical_exposure": "All source rows have previously been read and some used in auxiliary training; this is a common-data internal repartition, not a new external blind test",
    })

    fixture_manifest = load(FIXTURE / "PREPARATION_MANIFEST.json")
    conversion_manifest = load(CONVERTED / "manifest.json")
    source_hashes, written = {}, Counter()
    source_specs = [
        (FIXTURE / "inputs/answers.jsonl", "inputs.jsonl", True),
        (FIXTURE / "geometry/sequences.jsonl", "geometry.jsonl", True),
        (FIXTURE / "geometry/windows.jsonl", "window_geometry.jsonl", True),
        (FIXTURE / "scoring/answers.jsonl", "scoring.jsonl", False),
        *[(CONVERTED / name, name, False) for name in ("answers.jsonl", "sequences.jsonl", "tokens.jsonl", "windows.jsonl")],
    ]
    with ExitStack() as stack:
        handles = {}
        def get_handle(relative):
            if relative not in handles:
                path = OUT / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                handles[relative] = stack.enter_context(path.open("w", encoding="utf-8", newline="\n"))
            return handles[relative]
        for source, basename, safe in source_specs:
            sha = file_sha(source)
            if source.is_relative_to(FIXTURE):
                assert sha == fixture_manifest["files_sha256"][source.relative_to(FIXTURE).as_posix()]
            else:
                assert sha == conversion_manifest["files"][basename]["sha256"]
            source_hashes[str(source.relative_to(PROJECT)).replace("\\", "/")] = sha
            for row, original_line in lines(source):
                assignment = by_id[row["response_id"]]
                split = assignment["split"]
                if safe:
                    assert_safe(row)
                    relative = f"{split}/{basename}"
                    get_handle(relative).write(original_line)
                elif basename == "scoring.jsonl":
                    relative = f"{split}/sealed/labels.jsonl" if split == "test" else f"{split}/scoring.jsonl"
                    get_handle(relative).write(original_line)
                elif split == "test":
                    # Sealed scoring records already retain original spans and all token/window targets.
                    continue
                else:
                    relative = f"{split}/{basename}"
                    row["legacy_group_id"] = row["group_id"]
                    row["group_id"] = assignment["group_id"]
                    row["partition"] = split
                    row["no_hint"] = assignment["no_hint"]
                    emit(get_handle(relative), row)
                written[relative] += 1
    for split, count in counts.items():
        for name in ("inputs.jsonl", "geometry.jsonl"):
            assert written[f"{split}/{name}"] == count["answers"]
        if split != "test":
            assert written[f"{split}/answers.jsonl"] == written[f"{split}/sequences.jsonl"] == count["answers"]
    assert written["test/sealed/labels.jsonl"] == counts["test"]["answers"]
    # Verify raw/clipped geometry and original strings on the physical predictor exports only.
    geometry_checks = 0
    for split in counts:
        for (visible, _), (geometry, _) in zip(lines(OUT / split / "inputs.jsonl"),
                                               lines(OUT / split / "geometry.jsonl"), strict=True):
            assert visible["response_id"] == geometry["response_id"]
            assert visible["prompt_prefix"] == visible["released_prompt"] + " "
            assert digest(visible) == geometry["input_sha256"]
            assert digest(visible["prompt_prefix"] + visible["original_response"]) == geometry["rendered_input_sha256"]
            assert geometry["answer_character_range"][0] == len(visible["prompt_prefix"])
            assert geometry["context_character_range"] == visible["document_char_spans"][0]
            start, end = geometry["context_character_range"]
            assert visible["prompt_prefix"][start:end] == visible["retrieved_passages"]
            assert [geometry["full_input_ids"][i] for i in geometry["answer_token_positions"]] == geometry["response_token_ids"]
            text = visible["original_response"]
            for clipped, raw in zip(geometry["response_token_offsets"], geometry["response_token_offsets_raw"], strict=True):
                assert clipped == [max(0, raw[0]), min(len(text), raw[1])]
            geometry_checks += 1
    (OUT / "test/sealed/README.md").write_text(
        "# Sealed internal test targets\n\nOnly the final scorer may read labels.jsonl after model, threshold, and feature selection are frozen. "
        "It contains copied automatic span/token/window labels. No test label statistics were computed by this preparation. "
        "The underlying official train data has prior research exposure; do not call this an unseen external test.\n", encoding="utf-8")
    files = {str(path.relative_to(OUT)).replace("\\", "/"): {"sha256": file_sha(path), "bytes": path.stat().st_size}
             for path in sorted(OUT.rglob("*")) if path.is_file()}
    manifest = {
        "schema_version": VERSION, "complete": True, "seed": SEED,
        "dataset": "F4biian/RAGognize", "revision": "aab54518c2a7c0d25fff8bffbf5337d0321de142",
        "official_source_split": "train", "generator": "Llama-2-7b-chat-hf", "counts": counts,
        "split_assignment_sha256": file_sha(OUT / "split_assignments.jsonl"),
        "groups": audit, "source_files_sha256": source_hashes, "files": files,
        "geometry_records_verified": geometry_checks,
        "predictor_exports_reused_verbatim": True,
        "test_label_statistics_computed": False, "test_labels_physically_separate": True,
        "training_cohort": "all fit answers", "primary_evaluation_cohort": "no_hint",
        "secondary_evaluation_cohort": "all", "annotation_type": "automatic silver character spans",
        "comparison_requirement": "All methods use these exact fit/validation/test memberships; initialize from generic base models and new heads, never the official RAGognizer checkpoint trained on all official train",
        "historical_exposure": "Common-data internal repartition; official train has been read and used in prior research. Official test2781 is already seen and excluded.",
        "GPU_used": False, "model_forward": False,
        "builder_sha256": file_sha(Path(__file__)),
    }
    save(OUT / "MANIFEST.json", manifest)
    print(json.dumps({"output": str(OUT), "counts": counts, "group_audit": audit,
                      "geometry_records_verified": geometry_checks, "test_label_statistics_computed": False}, indent=2))


if __name__ == "__main__":
    main()
