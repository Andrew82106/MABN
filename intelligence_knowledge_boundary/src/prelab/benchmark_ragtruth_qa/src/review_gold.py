"""Independent character-enumeration audit; only frozen fit/calibration inputs.

Does not import build_gold, tokenize text, load a model, or read predictions/test.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
REPORT = DATA / "gold_independent_review.json"
EXPECTED_GOLD_MANIFEST_SHA = "e4e192e0c24339e89135390fcfb1bfcfb69b54d824c3803e2efb9b494c7546b7"


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def intervals(indices):
    out = []
    for i in sorted(set(indices)):
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


def oracle(text, offsets, labels):
    # Deliberately enumerate characters rather than test interval overlaps.
    char_memberships = [set() for _ in text]
    for j, (lo, hi) in enumerate(offsets):
        for i in range(lo, hi):
            char_memberships[i].add(j)
    risky_chars = set()
    span_tokens = []
    anomalies = []
    for k, label in enumerate(labels):
        lo, hi = label["start"], label["end"]
        assert 0 <= lo <= hi <= len(text), ("invalid_span", k)
        assert text[lo:hi] == label["text"], ("span_text_mismatch", k)
        members = set()
        alnum_chars = []
        for i in range(lo, hi):
            if text[i].isalnum():
                risky_chars.add(i)
                alnum_chars.append(i)
                members.update(char_memberships[i])
        span_tokens.append(sorted(members))
        flags = []
        if lo == hi:
            flags.append("zero_length_span")
        if not alnum_chars:
            flags.append("span_without_isalnum")
        uncovered = [i for i in range(lo, hi) if not char_memberships[i]]
        missing_alnum = [i for i in alnum_chars if not char_memberships[i]]
        if uncovered:
            flags.append("span_uncovered_characters")
        if missing_alnum:
            flags.append("span_uncovered_isalnum")
        if flags:
            anomalies.append({"span_index": k, "span_range": [lo, hi],
                              "flags": flags, "uncovered_character_ranges": intervals(uncovered),
                              "uncovered_isalnum_ranges": intervals(missing_alnum)})
    lexical, risk, token_risk_chars = [], [], []
    for lo, hi in offsets:
        chars = [i for i in range(lo, hi) if i in risky_chars]
        lexical.append(int(any(text[i].isalnum() for i in range(lo, hi))))
        risk.append(int(bool(chars)))
        token_risk_chars.append(chars)
    n = len(offsets)
    starts = range(n - 3) if n >= 4 else ([0] if n else [])
    windows = []
    for start in starts:
        stop = min(start + 4, n)
        covered = set()
        risk_chars = set()
        for j in range(start, stop):
            covered.update(range(*offsets[j]))
            risk_chars.update(token_risk_chars[j])
        windows.append({"start": start, "end": stop,
                        "eligible": bool(any(lexical[start:stop])),
                        "gold": int(any(risk[start:stop])),
                        "risk_token_indices": [j for j in range(start, stop) if risk[j]],
                        "lexical_token_indices": [j for j in range(start, stop) if lexical[j]],
                        "character_ranges": intervals(covered),
                        "risk_character_ranges": intervals(risk_chars),
                        "char_start": min(covered) if covered else 0,
                        "char_end": max(covered) + 1 if covered else 0})
    answer_risk = int(bool(labels))
    if answer_risk and not any(risk):
        anomalies.append({"flags": ["positive_answer_without_risk_tokens"]})
    return {"lexical_mask": lexical, "risk_mask": risk,
            "token_risk_character_ranges": [intervals(x) for x in token_risk_chars],
            "risk_character_ranges": intervals(risky_chars),
            "span_risk_token_indices": span_tokens, "answer_risk": answer_risk,
            "windows": windows, "anomalies": anomalies}


def synthetic_checks():
    # Expected values were independently hand-derived, including punctuation,
    # boundary crossing, repeated offsets, and response/window disagreement.
    cases = [
        ("same_character_conjunction", "A!", [[0, 2]], [(1, 2)], [1], [0], [(0, 1, True, 0)]),
        ("negative_raw_first_offset", "Cat", [[0, 3]], [(0, 1)], [1], [1], [(0, 1, True, 1)]),
        ("raw_punctuation_windows", "A,B.C", [[i, i+1] for i in range(5)], [(4, 5)], [1, 0, 1, 0, 1], [0, 0, 0, 0, 1], [(0, 4, True, 0), (1, 5, True, 1)]),
        ("single_short_window", "Hi!", [[i, i+1] for i in range(3)], [(1, 2)], [1, 1, 0], [0, 1, 0], [(0, 3, True, 1)]),
        ("nonlexical_window_excluded", "!? .", [[i, i+1] for i in range(4)], [], [0]*4, [0]*4, [(0, 4, False, 0)]),
        ("unlabelled_refusal_negative", "I cannot answer.", [[0, 1], [1, 8], [8, 15], [15, 16]], [], [1, 1, 1, 0], [0]*4, [(0, 4, True, 0)]),
        ("overlapping_offsets_retained", "猫A!", [[0, 1], [0, 1], [0, 1], [1, 2], [2, 3]], [(0, 1)], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [(0, 4, True, 1), (1, 5, True, 1)]),
        ("zero_length_positive_answer", "AB", [[0, 1], [1, 2]], [(1, 1)], [1, 1], [0, 0], [(0, 2, True, 0)]),
    ]
    passed = []
    for name, text, offsets, spans, lexical, risk, windows in cases:
        labels = [{"start": a, "end": b, "text": text[a:b]} for a, b in spans]
        got = oracle(text, offsets, labels)
        assert got["lexical_mask"] == lexical, name
        assert got["risk_mask"] == risk, name
        assert got["answer_risk"] == int(bool(labels)), name
        actual_windows = [(w["start"], w["end"], w["eligible"], w["gold"]) for w in got["windows"]]
        assert actual_windows == windows, name
        if name == "negative_raw_first_offset":
            full_offsets, answer_range = [[0, 4]], [1, 4]
            selected = [i for i, (a, b) in enumerate(full_offsets)
                        if max(a, answer_range[0]) < min(b, answer_range[1])]
            assert selected == [0]
            assert [[a-answer_range[0], b-answer_range[0]] for a, b in full_offsets] == [[-1, 3]]
        if name == "overlapping_offsets_retained":
            assert got["span_risk_token_indices"] == [[0, 1, 2]]
            assert got["risk_character_ranges"] == [[0, 1]]
        passed.append(name)
    return passed


def build_oracles():
    development = {}
    for partition, expected in [("fit", 634), ("calibration", 159)]:
        rows = list(read_rows(DATA / f"{partition}.jsonl"))
        assert len(rows) == expected
        for row in rows:
            rid = row["response_id"]
            assert rid not in development
            assert row["partition"] == partition and row["official_split"] == "train"
            assert row["quality"] == "good" and row["model"] == "llama-2-7b-chat"
            assert hashlib.sha256(row["original_response"].encode()).hexdigest() == row["answer_sha256"]
            development[rid] = row
    result, all_anomalies = {}, []
    counts = {p: Counter() for p in ["fit", "calibration"]}
    for plan in read_rows(DATA / "feature_preparation/plans.jsonl"):
        rid = plan["response_id"]
        assert rid in development and rid not in result
        row = development[rid]
        for key in ["source_id", "group_id", "partition", "official_split", "original_response", "answer_sha256", "prompt_sha256"]:
            assert row[key] == plan[key], (rid, key)
        original = plan["original"]  # No access to the no_context branch.
        text = row["original_response"]
        answer_lo, answer_hi = original["answer_character_range"]
        assert answer_hi-answer_lo == len(text)
        assert original["prefix_character_length"] == answer_lo
        full_offsets = original["input_token_offsets"]
        assert len(full_offsets) == len(original["input_ids"]) == len(original["attention_mask"])
        selected = [j for j, (lo, hi) in enumerate(full_offsets)
                    if max(lo, answer_lo) < min(hi, answer_hi)]
        raw = [[full_offsets[j][0]-answer_lo, full_offsets[j][1]-answer_lo] for j in selected]
        clipped = [[max(0, lo), min(len(text), hi)] for lo, hi in raw]
        ids = [original["input_ids"][j] for j in selected]
        assert selected == original["answer_token_positions"], (rid, "answer_token_selection")
        assert ids == original["answer_token_ids"], (rid, "answer_token_ids")
        assert raw == original["response_token_offsets_raw"], (rid, "raw_offsets")
        assert clipped == original["response_token_offsets"], (rid, "clipped_offsets")
        got = oracle(text, clipped, row["labels"])
        got.update(source=row, ids=ids, positions=selected, raw_offsets=raw, offsets=clipped)
        result[rid] = got
        c = counts[row["partition"]]
        eligible = [w for w in got["windows"] if w["eligible"]]
        c.update(answers=1, positive_answers=got["answer_risk"], raw_tokens=len(ids),
                 lexical_tokens=sum(got["lexical_mask"]), risk_tokens=sum(got["risk_mask"]),
                 raw_windows=len(got["windows"]), eligible_windows=len(eligible),
                 positive_windows=sum(w["gold"] for w in eligible),
                 excluded_nonlexical_windows=len(got["windows"])-len(eligible),
                 original_spans=len(row["labels"]),
                 negative_first_raw_offset_answers=int(bool(raw and raw[0][0] < 0)),
                 short_answer_windows=int(0 < len(ids) < 4),
                 repeated_offsets=len(clipped)-len(set(map(tuple, clipped))))
        for anomaly in got["anomalies"]:
            all_anomalies.append({"response_id": rid, "partition": row["partition"], **anomaly})
    assert result.keys() == development.keys()
    return result, {p: dict(c) for p, c in counts.items()}, all_anomalies


def span_objects(text, ranges):
    return [{"start": lo, "end": hi, "text": text[lo:hi]} for lo, hi in ranges]


def compare_exports(expected, counts):
    manifest_path = DATA / "gold_manifest.json"
    assert sha(manifest_path) == EXPECTED_GOLD_MANIFEST_SHA, "Gold manifest differs from producer handoff"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["signature"]["window_k"] == 4 and manifest["signature"]["window_stride"] == 1
    for key in ["label_types_filtered", "refusal_classifier_used", "tokenizer_rerun"]:
        assert manifest["signature"][key] is False
    sources = {
        "annotation_protocol": BASE / "ANNOTATION_PROTOCOL.md",
        "build_gold_code": BASE / "src/build_gold.py",
        "development_manifest": DATA / "development_manifest.json",
        "feature_preparation_manifest": DATA / "feature_preparation/manifest.json",
        "feature_preparation_plans": DATA / "feature_preparation/plans.jsonl",
        "fit_source": DATA / "fit.jsonl", "calibration_source": DATA / "calibration.jsonl",
    }
    assert manifest["source_paths"].keys() == sources.keys()
    for key, path in sources.items():
        # Only open this explicit trusted allowlist, not arbitrary manifest paths.
        assert Path(manifest["source_paths"][key]).resolve() == path.resolve()
        assert sha(path) == manifest["signature"]["files_sha256"][key], ("source_hash", key)
    answer_ids, window_ids = {}, set()
    row_counts, export_hashes = {}, {}

    def compare_fields(actual, wanted, tag):
        for key, value in wanted.items():
            assert actual[key] == value, (tag, key)

    def identity(row, partition):
        rid = row["response_id"]
        assert rid in expected, ("unknown_response", rid)
        source = expected[rid]["source"]
        assert source["partition"] == partition
        compare_fields(row, {k: source[k] for k in ["response_id", "source_id", "group_id", "partition"]}, rid)
        return rid, expected[rid], source

    groups = {}
    for partition in ["fit", "calibration"]:
        partition_ids = {rid for rid, got in expected.items() if got["source"]["partition"] == partition}
        groups[partition] = {expected[rid]["source"]["group_id"] for rid in partition_ids}
        seen = set()
        for row in read_rows(DATA / f"tokens_{partition}.jsonl"):
            rid, got, source = identity(row, partition)
            assert rid not in seen
            seen.add(rid)
            answer_ids[rid] = row["answer_id"]
            text = source["original_response"]
            compare_fields(row, {
                "original_response": text, "answer_sha256": source["answer_sha256"],
                "original_labels": source["labels"], "token_count": len(got["ids"]),
                "token_ids": got["ids"], "answer_token_positions": got["positions"],
                "response_token_offsets_raw": got["raw_offsets"], "response_token_offsets": got["offsets"],
                "lexical_mask": got["lexical_mask"], "risk_mask": got["risk_mask"],
                "risk_character_spans": span_objects(text, got["risk_character_ranges"]),
                "token_risk_character_spans": [span_objects(text, ranges) for ranges in got["token_risk_character_ranges"]],
                "span_token_mapping": [{"span_index": k, "original_start": label["start"], "original_end": label["end"],
                                        "risk_token_indices": got["span_risk_token_indices"][k]} for k, label in enumerate(source["labels"])],
                "answer_risk": got["answer_risk"], "first_answer_token_preserved": True,
                "edge_case_flags": [],
            }, (rid, "tokens"))
        assert seen == partition_ids
        row_counts[f"tokens_{partition}.jsonl"] = len(seen)

        seen_windows = {rid: set() for rid in partition_ids}
        for filename, eligible in [(f"windows_k4_{partition}.jsonl", True), (f"windows_excluded_{partition}.jsonl", False)]:
            num_rows = 0
            for row in read_rows(DATA / filename):
                rid, got, source = identity(row, partition)
                start = row["token_start"]
                assert 0 <= start < len(got["windows"]), (rid, "invalid_window_start", start)
                assert start not in seen_windows[rid], (rid, "duplicate_window_start", start)
                seen_windows[rid].add(start)
                window = got["windows"][start]
                stop = window["end"]
                assert window["eligible"] is eligible
                assert row["window_id"] not in window_ids
                window_ids.add(row["window_id"])
                text = source["original_response"]
                compare_fields(row, {
                    "answer_id": answer_ids[rid], "k": 4, "stride": 1,
                    "token_start": start, "token_end": stop, "token_indices": list(range(start, stop)),
                    "answer_token_positions": got["positions"][start:stop], "token_ids": got["ids"][start:stop],
                    "character_intervals": window["character_ranges"],
                    "char_start": window["char_start"], "char_end": window["char_end"],
                    "bounding_text": text[window["char_start"]:window["char_end"]],
                    "lexical_token_indices": window["lexical_token_indices"],
                    "risk_token_indices": window["risk_token_indices"],
                    "risk_character_spans": span_objects(text, window["risk_character_ranges"]),
                    "eligible": eligible, "label": window["gold"] if eligible else None,
                }, (rid, start, "window"))
                if not eligible:
                    assert row["exclusion_reason"] == "no_lexical_token"
                    assert not window["lexical_token_indices"] and not window["risk_token_indices"]
                num_rows += 1
            row_counts[filename] = num_rows
        for rid in partition_ids:
            assert seen_windows[rid] == set(range(len(expected[rid]["windows"]))), (rid, "window_coverage")

        seen = set()
        for row in read_rows(DATA / f"answers_{partition}.jsonl"):
            rid, got, source = identity(row, partition)
            assert rid not in seen
            seen.add(rid)
            eligible_windows = [w for w in got["windows"] if w["eligible"]]
            compare_fields(row, {
                "answer_id": answer_ids[rid], "original_response": source["original_response"],
                "answer_sha256": source["answer_sha256"], "original_labels": source["labels"],
                "quality": "good", "eligible": True, "label": got["answer_risk"],
                "token_count": len(got["ids"]), "lexical_token_count": sum(got["lexical_mask"]),
                "risk_token_count": sum(got["risk_mask"]), "official_span_count": len(source["labels"]),
                "candidate_window_count": len(got["windows"]), "eligible_window_count": len(eligible_windows),
                "positive_window_count": sum(w["gold"] for w in eligible_windows),
                "excluded_window_count": len(got["windows"])-len(eligible_windows), "edge_case_flags": [],
            }, (rid, "answer"))
        assert seen == partition_ids
        row_counts[f"answers_{partition}.jsonl"] = len(seen)
    assert len(set(answer_ids.values())) == len(expected)
    assert not groups["fit"].intersection(groups["calibration"])

    selfcheck = json.loads((DATA / "gold_selfcheck.json").read_text(encoding="utf-8"))
    assert selfcheck["passed"] is True
    assert selfcheck["answers_checked"] == len(expected)
    assert selfcheck["raw_tokens_checked"] == sum(c["raw_tokens"] for c in counts.values())
    assert selfcheck["original_spans_checked"] == sum(c["original_spans"] for c in counts.values())
    assert selfcheck["first_boundary_crossing_tokens_preserved"] == sum(c["negative_first_raw_offset_answers"] for c in counts.values())
    for partition, c in counts.items():
        wanted = {"answers": c["answers"], "positive_answers": c["positive_answers"],
                  "negative_answers": c["answers"]-c["positive_answers"],
                  "raw_tokens": c["raw_tokens"], "lexical_tokens": c["lexical_tokens"], "risk_tokens": c["risk_tokens"],
                  "official_spans": c["original_spans"], "candidate_windows": c["raw_windows"],
                  "eligible_windows": c["eligible_windows"], "positive_windows": c["positive_windows"],
                  "negative_windows": c["eligible_windows"]-c["positive_windows"],
                  "excluded_no_lexical_windows": c["excluded_nonlexical_windows"], "groups": len(groups[partition])}
        assert manifest["counts"][partition] == wanted
        assert selfcheck["counts"][partition] == wanted
    edges = json.loads((DATA / "gold_edge_cases.json").read_text(encoding="utf-8"))
    assert all(not got["anomalies"] for got in expected.values())
    assert all(any(w["eligible"] for w in got["windows"]) for got in expected.values())
    assert all(value == 0 for value in edges["counts"].values())
    assert all(value == [] for value in edges["cases"].values())
    assert manifest["edge_case_counts"] == edges["counts"]
    allowed_outputs = {DATA / name for name in row_counts} | {DATA / "gold_edge_cases.json", DATA / "gold_selfcheck.json"}
    assert len(manifest["outputs"]) == len(allowed_outputs)
    assert {(BASE / item["path"]).resolve() for item in manifest["outputs"]} == {p.resolve() for p in allowed_outputs}
    for item in manifest["outputs"]:
        path = (BASE / item["path"]).resolve()
        assert path.stat().st_size == item["bytes"]
        digest = sha(path)
        assert digest == item["sha256"], ("export_hash", path.name)
        if path.name in row_counts:
            assert row_counts[path.name] == item["rows"]
        export_hashes[path.name] = digest
    assert sha(manifest_path) == EXPECTED_GOLD_MANIFEST_SHA, "Gold manifest changed during audit"
    return {"gold_manifest_sha256": EXPECTED_GOLD_MANIFEST_SHA, "export_sha256": export_hashes,
            "export_rows_checked": row_counts, "full_input_token_reselection_exact": True,
            "all_token_arrays_character_ranges_and_span_mappings_exact": True,
            "all_raw_windows_and_exclusions_exact": True, "all_answer_labels_and_counts_exact": True,
            "all_unlabelled_answers_and_their_lexical_windows_retained_negative": True,
            "fit_calibration_groups_disjoint": True, "all_source_and_output_manifest_hashes_match": True,
            "producer_selfcheck_counts_independently_confirmed": True}


def main():
    report = {"reviewer": "/root/data_build/extract_review", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "scope": "Frozen fit634/calibration159 only; no test/withheld/predictions/model/GPU",
              "oracle_method": "Independent per-Unicode-character memberships; full-input offset token reselection; no import of build_gold",
              "input_sha256": {str(p.relative_to(BASE)): sha(p) for p in [BASE / "ANNOTATION_PROTOCOL.md", DATA / "fit.jsonl", DATA / "calibration.jsonl", DATA / "feature_preparation/plans.jsonl"]}}
    try:
        report["synthetic_checks_passed"] = synthetic_checks()
        expected, counts, anomalies = build_oracles()
        report.update(status="oracle_complete_exports_pending", answer_count=len(expected), counts=counts,
                      anomalies=anomalies, anomaly_flag_counts=dict(Counter(flag for a in anomalies for flag in a["flags"])))
        if (DATA / "gold_manifest.json").exists():
            report["export_comparison"] = compare_exports(expected, counts)
            report.update(status="passed", completed_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error_detail=str(error))
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    report["review_script_sha256"] = sha(Path(__file__))
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["status", "answer_count", "counts", "anomaly_flag_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
