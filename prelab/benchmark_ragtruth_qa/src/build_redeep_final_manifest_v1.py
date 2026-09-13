"""Build the final hash manifest for the completed ReDeEP calibration migration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "research" / "redeep_formal_baseline_v1"
OUT = ROOT / "FINAL_ARTIFACT_MANIFEST.json"
FILES = [
    PROJECT / "src" / name
    for name in (
        "audit_redeep_raw_features_v1.py",
        "build_redeep_final_manifest_v1.py",
        "prepare_redeep_formal_baseline_v1.py",
        "redeep_formal_core_v1.py",
        "run_redeep_formal_baseline_v1.py",
        "score_redeep_formal_baseline_v1.py",
        "selftest_redeep_formal_baseline_v1.py",
        "verify_redeep_extraction_identity_v1.py",
        "verify_redeep_scoring_v1.py",
    )
] + [
    ROOT / name
    for name in (
        "CACHE_REUSE_AUDIT.json",
        "CAL_RESULTS.json",
        "CPU_SELFTEST.json",
        "DATA_JOIN_AUDIT.json",
        "evaluation_geometry.jsonl",
        "EXTRACTION_IDENTITY_VERIFICATION.json",
        "feature_inputs.jsonl",
        "frozen_scores.npz",
        "INCIDENT_OFFICIAL_TEST_SEAL.md",
        "METHOD_FREEZE_PRE_EXTRACTION.json",
        "METHOD_FREEZE.json",
        "RAW_FEATURE_COMPLETE.json",
        "RAW_FEATURE_MANIFEST.json",
        "REPORT.md",
        "RESOURCE_ESTIMATE.json",
        "RUN_IDENTITY_EVIDENCE.json",
        "RUNBOOK.md",
        "SCORE_FREEZE.json",
        "SCORING_INDEPENDENT_VERIFICATION.json",
        "SCORING_STATUS.json",
        "SOURCE_AUDIT.json",
    )
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def main() -> None:
    raw = load_json("RAW_FEATURE_COMPLETE.json")
    extraction = load_json("EXTRACTION_IDENTITY_VERIFICATION.json")
    score_check = load_json("SCORING_INDEPENDENT_VERIFICATION.json")
    results = load_json("CAL_RESULTS.json")
    status = load_json("SCORING_STATUS.json")
    if not (
        raw["status"] == "complete"
        and raw["files"] == 3839
        and raw["native_tokens"] == 710970
        and extraction["status"] == "passed"
        and extraction["representative_rows"] == 4
        and score_check["status"] == "passed"
        and results["status"] == "complete"
        and status["status"] == "calibration_complete_and_independently_verified"
    ):
        raise RuntimeError("a required completion gate is not satisfied")
    entries = []
    for path in FILES:
        entries.append({
            "path": path.relative_to(PROJECT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    payload = {
        "version": "redeep-formal-final-artifact-manifest-v1",
        "status": "calibration_complete_and_verified",
        "scope": "shared fit/calibration only; independent untouched holdout remains N/A",
        "entries": entries,
        "entry_count": len(entries),
        "raw_feature_payload": {
            "individual_files": 3839,
            "individual_file_hashes_in": "research/redeep_formal_baseline_v1/RAW_FEATURE_MANIFEST.json",
            "native_tokens": 710970,
        },
        "formal_primary": "paper-explicit ReDeEP(Token) identity; official-code identity is diagnostic only",
        "test_artifacts_accessed_by_this_stage": False,
        "historical_official_test_status": "retired_after_prior_documented_access_incident",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "entries": len(entries), "manifest_sha256": sha256_file(OUT)}))


if __name__ == "__main__":
    main()
