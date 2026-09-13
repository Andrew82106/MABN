"""Independent stdlib-only audit of the full RAGognizer inference plan."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLAN = HERE / "INFERENCE_PLAN.jsonl"
OUTPUT = HERE / "PLAN_INDEPENDENT_AUDIT.json"
EXPECTED_SHA256 = "90d21b3bc9aa8e92dc7c6ef8fac3d7e178effa0236a5464831541ea401343a76"
EXPECTED_ROWS = {"fit": 3_680, "calibration": 159}
EXPECTED_GROUPS = {"fit": 615, "calibration": 154}
EXPECTED_WINDOWS = {"fit": 653_979, "calibration": 42_241}
EXPECTED_LOCAL_SLOTS = {"fit": 665_708, "calibration": 42_798}
EXPECTED_EXACT_ROWS = {"fit": 1_479, "calibration": 20}
WINDOW_RE = re.compile(r"__k4_(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def merge(intervals):
    result = []
    for start, end in sorted(
        (int(a), int(b)) for a, b in intervals if int(b) > int(a)
    ):
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return result


def check_no_forbidden_keys(value: object, location: str) -> None:
    forbidden = ("label", "risk", "gold", "annotation")
    if isinstance(value, dict):
        for key, child in value.items():
            if any(term in str(key).lower() for term in forbidden):
                raise RuntimeError(f"forbidden plan key {location}.{key}")
            check_no_forbidden_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_no_forbidden_keys(child, f"{location}[{index}]")


def main() -> None:
    if EXPECTED_SHA256 == "__FULL_PLAN_SHA256__":
        raise RuntimeError("independent audit is not frozen: plan hash placeholder remains")
    if sha256_file(PLAN) != EXPECTED_SHA256:
        raise RuntimeError("plan hash changed")
    if any(not isinstance(value, int) for value in EXPECTED_LOCAL_SLOTS.values()):
        raise RuntimeError("independent audit local-slot placeholders remain")
    if any(not isinstance(value, int) for value in EXPECTED_EXACT_ROWS.values()):
        raise RuntimeError("independent audit exact-row placeholders remain")

    answer_ids = set()
    groups: dict[str, set[str]] = defaultdict(set)
    window_ids = set()
    row_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    local_slots: Counter[str] = Counter()
    native_tokens: Counter[str] = Counter()
    exact_rows: Counter[str] = Counter()
    mapping_rows: Counter[str] = Counter()
    coordinate_modes: Counter[str] = Counter()
    coordinate_fallback_ids: list[str] = []
    short_window_rows: list[dict] = []
    for row_index, row in enumerate(
        json.loads(line)
        for line in PLAN.open("r", encoding="utf-8")
        if line.strip()
    ):
        check_no_forbidden_keys(row, f"row[{row_index}]")
        if row.get("version") != "ragognizer-full-inference-plan-v1":
            raise RuntimeError("plan version changed")
        stored_row_hash = str(row.get("plan_row_sha256", ""))
        core = {key: value for key, value in row.items() if key != "plan_row_sha256"}
        if stored_row_hash != sha256_json(core):
            raise RuntimeError(f"plan row hash changed at {row_index}")
        partition = str(row.get("partition"))
        if partition not in EXPECTED_ROWS or row.get("official_split") != "train":
            raise RuntimeError("partition or official split boundary changed")
        answer_id = str(row["answer_id"])
        if answer_id in answer_ids:
            raise RuntimeError("duplicate answer")
        answer_ids.add(answer_id)
        row_counts[partition] += 1
        groups[partition].add(str(row["group_id"]))
        if row["chat"] != [
            {"role": "user", "content": row["released_prompt"]},
            {"role": "assistant", "content": row["original_response"]},
        ]:
            raise RuntimeError("chat construction changed")

        native = row["author_response_tokens"]
        local = row["shared_bpe_tokens"]
        probability_indices = list(map(int, native["probability_indices"]))
        coverage = [False] * len(row["original_response"])
        for start, end in native["char_intervals"]:
            coverage[int(start) : int(end)] = [True] * (int(end) - int(start))
        if not all(coverage) or native["response_character_coverage"]["uncovered"] != 0:
            raise RuntimeError(f"author coordinate coverage differs for {answer_id}")
        coordinate_modes[native["coordinate_mode"]] += 1
        if native["official_pack_exact"]:
            if "".join(native["coordinate_texts"]) != row["original_response"]:
                raise RuntimeError(f"packed author text differs for {answer_id}")
            if native["official_pack_failure_sha256"] is not None:
                raise RuntimeError(f"ordinary row carries pack failure for {answer_id}")
        else:
            coordinate_fallback_ids.append(answer_id)
            byte_indices = list(map(int, native["byte_fallback_local_indices"]))
            probability_byte_indices = list(
                map(int, native["byte_fallback_probability_indices"])
            )
            if not byte_indices or [probability_indices[index] for index in byte_indices] != probability_byte_indices:
                raise RuntimeError(f"byte-fallback index proof differs for {answer_id}")
            if native["official_pack_failure_sha256"] is None:
                raise RuntimeError(f"byte-fallback failure hash missing for {answer_id}")
        if probability_indices != sorted(set(probability_indices)):
            raise RuntimeError(
                f"author probability indices are not strictly increasing for {answer_id}"
            )
        model_ids = row["author_roundtrip"]["model_input_ids"]
        if [model_ids[index] for index in probability_indices] != native["token_ids"]:
            raise RuntimeError(f"author token/probability index mismatch for {answer_id}")

        expected_map = []
        if row["mapping"]["mode"] == "exact_token":
            if native["token_ids"] != local["token_ids"]:
                raise RuntimeError(f"false exact token identity for {answer_id}")
            if native["char_intervals"] != local["char_intervals"]:
                raise RuntimeError(f"false exact interval identity for {answer_id}")
            expected_map = [[index] for index in range(len(local["token_ids"]))]
        elif row["mapping"]["mode"] == "character_overlap":
            for local_start, local_end in local["char_intervals"]:
                overlap = [
                    index
                    for index, (native_start, native_end) in enumerate(
                        native["char_intervals"]
                    )
                    if max(int(local_start), int(native_start))
                    < min(int(local_end), int(native_end))
                ]
                if not overlap:
                    raise RuntimeError(f"uncovered local BPE for {answer_id}")
                expected_map.append(overlap)
        else:
            raise RuntimeError(f"unknown mapping mode for {answer_id}")
        if expected_map != row["mapping"]["local_to_author_token_indices"]:
            raise RuntimeError(f"stored token map differs for {answer_id}")
        mapping_rows[row["mapping"]["mode"]] += 1
        local_slots[partition] += len(expected_map)
        native_tokens[partition] += len(native["token_ids"])
        row_exact = (
            native["token_ids"] == local["token_ids"]
            and native["char_intervals"] == local["char_intervals"]
        )
        if row_exact != bool(row["mapping"]["row_tokens_happen_to_be_exact"]):
            raise RuntimeError(f"row exactness flag differs for {answer_id}")
        exact_rows[partition] += int(row_exact)

        previous = -1
        for window in row["eligible_windows"]:
            window_id = str(window["window_id"])
            if window_id in window_ids:
                raise RuntimeError("duplicate window")
            window_ids.add(window_id)
            match = WINDOW_RE.search(window_id)
            start = int(window["token_start"])
            if match is None or int(match.group(1)) != start or start <= previous:
                raise RuntimeError(f"window index/order changed for {window_id}")
            previous = start
            slot_count = int(window.get("slot_count", 0))
            local_count = len(local["token_ids"])
            expected_slot_count = 4 if local_count >= 4 else local_count
            if slot_count != expected_slot_count:
                raise RuntimeError(f"window slot count changed for {window_id}")
            if local_count < 4 and (start != 0 or len(row["eligible_windows"]) != 1):
                raise RuntimeError(f"short-window rule changed for {window_id}")
            if start + slot_count > local_count:
                raise RuntimeError(f"window exceeds local tokens for {window_id}")
            if merge(local["char_intervals"][start : start + slot_count]) != window[
                "character_intervals"
            ]:
                raise RuntimeError(f"window intervals changed for {window_id}")
            window_counts[partition] += 1
        if len(local["token_ids"]) < 4:
            short_window_rows.append({
                "answer_id": answer_id,
                "partition": partition,
                "shared_bpe_tokens": len(local["token_ids"]),
            })

    if dict(row_counts) != EXPECTED_ROWS:
        raise RuntimeError(f"answer counts changed: {dict(row_counts)}")
    if {key: len(groups[key]) for key in EXPECTED_GROUPS} != EXPECTED_GROUPS:
        raise RuntimeError("group counts changed")
    if groups["fit"] & groups["calibration"]:
        raise RuntimeError("fit/calibration groups overlap")
    if dict(window_counts) != EXPECTED_WINDOWS:
        raise RuntimeError(f"window counts changed: {dict(window_counts)}")
    if dict(local_slots) != EXPECTED_LOCAL_SLOTS:
        raise RuntimeError(f"local BPE slot counts changed: {dict(local_slots)}")
    if dict(exact_rows) != EXPECTED_EXACT_ROWS:
        raise RuntimeError(f"exact-token row counts changed: {dict(exact_rows)}")
    if dict(mapping_rows) != {"character_overlap": sum(EXPECTED_ROWS.values())}:
        raise RuntimeError("frozen tokenizer identities no longer imply global character overlap")
    if coordinate_modes != Counter({
        "author_pack_exact_fast_positions_verified": 3_836,
        "fast_offsets_author_pack_byte_fallback": 3,
    }):
        raise RuntimeError(f"author coordinate mode counts changed: {dict(coordinate_modes)}")
    if coordinate_fallback_ids != ["15069", "12426", "15066"]:
        raise RuntimeError(f"author byte-fallback rows changed: {coordinate_fallback_ids}")
    if short_window_rows != [{
        "answer_id": "14641", "partition": "fit", "shared_bpe_tokens": 1
    }]:
        raise RuntimeError(f"short-window rows changed: {short_window_rows}")

    result = {
        "version": "ragognizer-full-plan-independent-audit-v1",
        "status": "pass",
        "plan_sha256": sha256_file(PLAN),
        "answers": sum(row_counts.values()),
        "partition_answers": dict(row_counts),
        "groups": {key: len(groups[key]) for key in EXPECTED_GROUPS},
        "fit_calibration_group_intersection": 0,
        "windows": sum(window_counts.values()),
        "partition_windows": dict(window_counts),
        "local_bpe_slots": sum(local_slots.values()),
        "partition_local_bpe_slots": dict(local_slots),
        "author_response_tokens": sum(native_tokens.values()),
        "partition_author_response_tokens": dict(native_tokens),
        "row_exact_tokenizations": sum(exact_rows.values()),
        "partition_row_exact_tokenizations": dict(exact_rows),
        "mapping_rows": dict(mapping_rows),
        "author_coordinate_modes": dict(coordinate_modes),
        "author_pack_byte_fallback_answer_ids": coordinate_fallback_ids,
        "short_window_rows": short_window_rows,
        "all_plan_row_hashes_recomputed": True,
        "all_ordinary_native_packed_text_reconstructs_original_response": True,
        "all_fallback_coordinate_unions_cover_original_response": True,
        "all_probability_indices_match_author_token_ids": True,
        "all_stored_maps_recomputed_exactly": True,
        "all_window_geometry_recomputed_exactly": True,
        "labels_read": False,
        "gpu_workload_started": False,
        "test_opened": False,
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
