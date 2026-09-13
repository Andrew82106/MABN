"""Create a hash-bound, label-free manifest for complete ReDeEP raw features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from score_redeep_formal_baseline_v1 import (
    EXPECTED_GEOMETRY_SHA,
    EXPECTED_INPUT_SHA,
    GEOMETRY,
    INPUTS,
    PROJECT,
    ROOT,
    input_rows,
    raw_path,
    validate_raw,
)


MANIFEST = ROOT / "RAW_FEATURE_MANIFEST.json"
COMPLETE = ROOT / "RAW_FEATURE_COMPLETE.json"
CANDIDATE_HEADS_SHA = "633a37382dfe46242028cea3d4f28c6bb2102da1859572f6077a39a612228a4b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = input_rows()
    expected_heads = np.asarray(sorted(json.loads(
        (PROJECT / "third_party" / "ReDEeP-ICLR" / "ReDeEP" / "log" / "test_llama2_7B" / "topk_heads.json")
        .read_text(encoding="utf-8")
    )), dtype=np.int16)
    entries = []
    totals = {
        "files": 0,
        "bytes": 0,
        "native_tokens": 0,
        "partition_files": {"fit": 0, "calibration": 0},
    }
    ranges = {
        name: [float("inf"), float("-inf")]
        for name in ("ecs_code", "ecs_paper", "pks_code", "pks_paper")
    }
    failures = []
    for row in rows:
        path = raw_path(row)
        valid, reason = validate_raw(path, row)
        if not valid:
            failures.append({"response_id": row["response_id"], "reason": reason})
            continue
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["candidate_heads"], expected_heads):
                failures.append({"response_id": row["response_id"], "reason": "candidate_head_identity"})
                continue
            n = int(data["ecs_code"].shape[0])
            for name in ranges:
                ranges[name][0] = min(ranges[name][0], float(data[name].min()))
                ranges[name][1] = max(ranges[name][1], float(data[name].max()))
        size = path.stat().st_size
        relative = path.relative_to(PROJECT).as_posix()
        entries.append({
            "response_id": row["response_id"],
            "answer_sha256": row["answer_sha256"],
            "partition": row["partition"],
            "file": relative,
            "bytes": size,
            "native_tokens": n,
            "sha256": sha256_file(path),
        })
        totals["files"] += 1
        totals["bytes"] += size
        totals["native_tokens"] += n
        totals["partition_files"][row["partition"]] += 1
    if failures or totals["files"] != 3839 or totals["native_tokens"] != 710970:
        raise RuntimeError(
            f"raw feature set is incomplete/invalid: files={totals['files']}, tokens={totals['native_tokens']}, "
            f"failures={failures[:5]}"
        )
    manifest = {
        "version": "redeep-formal-raw-feature-manifest-v1",
        "status": "complete",
        "labels_read": False,
        "test_artifacts_accessed_by_this_stage": False,
        "historical_official_test_status": "retired_after_prior_documented_access_incident",
        "feature_inputs_sha256": EXPECTED_INPUT_SHA,
        "evaluation_geometry_sha256": EXPECTED_GEOMETRY_SHA,
        "candidate_heads_sha256": CANDIDATE_HEADS_SHA,
        "runner_sha256": sha256_file(PROJECT / "src" / "run_redeep_formal_baseline_v1.py"),
        "totals": totals,
        "global_ranges": ranges,
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    complete = {
        "version": "redeep-formal-raw-feature-complete-v1",
        "status": "complete",
        "manifest_sha256": sha256_file(MANIFEST),
        "files": totals["files"],
        "native_tokens": totals["native_tokens"],
        "all_records_validated": True,
        "all_values_finite": True,
        "labels_read": False,
        "test_artifacts_accessed_by_this_stage": False,
        "historical_official_test_status": "retired_after_prior_documented_access_incident",
    }
    COMPLETE.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(complete))


if __name__ == "__main__":
    main()
