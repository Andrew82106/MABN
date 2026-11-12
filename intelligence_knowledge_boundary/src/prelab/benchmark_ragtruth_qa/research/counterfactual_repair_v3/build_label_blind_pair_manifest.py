"""Materialize a physically label-free fit claim/source pair manifest.

Only explicit text, identity, and retrieval metadata fields are copied.  The
four upstream artifacts are themselves fit-only derived artifacts and carry no
QA correctness annotations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MICRO = ROOT / "research/atomic_relation_expanded_fit_v1/microclaims_fit3680.jsonl"
SEMANTIC = ROOT / "results/semantic_source_attribution_expanded_fit_v1/semantic_units.jsonl"
EVIDENCE = (
    ROOT / "results/evidence_union_nli_v2/candidates_fit_native.jsonl",
    ROOT / "results/evidence_union_nli_v2/candidates_fit_expanded.jsonl",
)
OUT = HERE / "input/claim_source_pairs_fit_label_blind.jsonl"
SUMMARY = HERE / "input/PAIR_MANIFEST.json"

EXPECTED = {
    MICRO: "c1731ab6ca68569f7811db46f594431751995d65d2468cdec51f8d63f37d585a",
    SEMANTIC: "eb3bbb3657a99c70ddde47427a3fcd45966e7f3ff7ef3e83f446d5afeb010820",
    EVIDENCE[0]: "129a11a87d4726724876761fdf3a4428f3f7365e6ccb4c05d5887d6ca8cbc84b",
    EVIDENCE[1]: "8d6ef184fc314561e47bc50e23fe0e036a3c389688f7468c425cc0e5199c675c",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    assert all(sha(path) == expected for path, expected in EXPECTED.items())

    microclaims = {}
    for row in rows(MICRO):
        assert row["partition"] == "fit"
        assert not ({"label", "labels", "gold", "error_type", "spans", "window_label"} & set(row))
        key = (str(row["response_id"]), str(row["microclaim_id"]))
        assert key not in microclaims
        microclaims[key] = {
            "response_id": str(row["response_id"]),
            "source_id": str(row["source_id"]),
            "group_id": str(row["group_id"]),
            "microclaim_id": str(row["microclaim_id"]),
            "microclaim_index": int(row["microclaim_index"]),
            "claim_start": int(row["start"]),
            "claim_end": int(row["end"]),
            "claim_text": str(row["text"]),
        }

    sources = {}
    for row in rows(SEMANTIC):
        assert row["partition"] == "fit"
        assert row["labels_used"] is False and row["official_test_opened"] is False
        source_id = str(row["source_id"])
        sentence_map = {
            (int(item["sentence_index"]), int(item["passage_id"]), int(item["sentence_id"]), str(item["text_sha256"])): str(item["text"])
            for item in row["sentences"]
        }
        if source_id in sources:
            assert sources[source_id] == sentence_map
        else:
            sources[source_id] = sentence_map

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(OUT.suffix + ".pending")
    seen, response_ids, claim_ids = set(), set(), set()
    pair_count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for evidence_path in EVIDENCE:
            for answer in rows(evidence_path):
                assert answer["partition"] == "fit"
                assert answer["labels_used"] is False and answer["official_test_opened"] is False
                assert not ({"label", "labels", "gold", "error_type", "spans", "window_label"} & set(answer))
                response_id = str(answer["response_id"])
                source_id = str(answer["source_id"])
                for claim in answer["claims"]:
                    key = (response_id, str(claim["microclaim_id"]))
                    meta = microclaims[key]
                    assert meta["source_id"] == source_id and meta["group_id"] == str(answer["group_id"])
                    response_ids.add(response_id)
                    claim_ids.add(key)
                    for candidate in claim["candidates"]:
                        sentence_key = (
                            int(candidate["sentence_index"]), int(candidate["passage_id"]),
                            int(candidate["sentence_id"]), str(candidate["sentence_sha256"]),
                        )
                        source_text = sources[source_id][sentence_key]
                        pair_id = "cerp3_pair_" + digest(
                            "|".join((response_id, meta["microclaim_id"], str(sentence_key[0]), str(sentence_key[1]), sentence_key[3]))
                        )[:20]
                        assert pair_id not in seen
                        seen.add(pair_id)
                        ranks = [value for value in (candidate.get("attention_rank"), candidate.get("bm25_rank")) if isinstance(value, int)]
                        output = {
                            **meta,
                            "pair_id": pair_id,
                            "sentence_index": sentence_key[0],
                            "passage_id": sentence_key[1],
                            "sentence_id": sentence_key[2],
                            "sentence_sha256": sentence_key[3],
                            "source_sentence": source_text,
                            "attention_rank": candidate.get("attention_rank"),
                            "bm25_rank": candidate.get("bm25_rank"),
                            "attention_scalar": candidate.get("attention_scalar"),
                            "bm25": candidate.get("bm25"),
                            "query_term_coverage": candidate.get("query_term_coverage"),
                            "union_rank": min(ranks) if ranks else 99,
                            "original_nli_request_id": candidate.get("request_id"),
                            "original_hypothesis_sha256": claim.get("hypothesis_sha256"),
                            "labels_used": False,
                        }
                        handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                        pair_count += 1
    tmp.replace(OUT)

    summary = {
        "status": "physically_label_blind_pair_manifest_complete",
        "partition": "fit",
        "pairs": pair_count,
        "claims": len(claim_ids),
        "responses": len(response_ids),
        "source_ids": len(sources),
        "selection_used_labels": False,
        "official_test_opened": False,
        "model_loaded": False,
        "gpu_used": False,
        "source_sha256": {str(path.relative_to(ROOT)): expected for path, expected in EXPECTED.items()},
        "artifact": str(OUT.relative_to(ROOT)),
        "artifact_sha256": sha(OUT),
    }
    atomic_json(SUMMARY, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
