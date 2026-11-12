"""Verify frozen RefChecker feasibility artifacts without model/test access."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / "third_party/reference/refchecker_1df1b25"


def read_json(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REF, text=True).strip()


def main() -> None:
    audit = read_json("CPU_AUDIT.json")
    metadata = read_json("MODEL_METADATA_SNAPSHOT.json")
    checklist = read_json("FEASIBILITY_CHECKLIST.json")
    report = (HERE / "REPORT.md").read_text(encoding="utf-8")

    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    tag = git("describe", "--tags", "--exact-match", "HEAD")
    clean = git("status", "--porcelain") == ""

    assert commit == "1df1b25cee792ba2b171302e31ca4f768bd67703"
    assert tree == "9f2e810cc91abaeeceb105cf4b8027eb463580f8"
    assert tag == "v0.2.17"
    assert clean
    assert "完整RefChecker正式基线当前不可运行" in report
    assert checklist["overall_status"] == (
        "blocked_complete_formal_baseline_not_runnable_on_local_8gib"
    )
    assert checklist["formal_score_may_be_reported_now"] is False
    assert checklist["dataset_binding"]["answers"] == 3839
    assert checklist["dataset_binding"]["fit_answers"] == 3680
    assert checklist["dataset_binding"]["eligible_4bpe_windows"] == 696220
    assert checklist["dataset_binding"]["excluded_no_lexical_windows"] == 772
    assert checklist["dataset_binding"]["ragtruth_human_spans"][
        "allowed_as_refchecker_claim_input"
    ] is False
    assert checklist["audit_guardrails"] == {
        "gpu_used": False,
        "weights_downloaded": False,
        "official_test_opened": False,
        "formal_baseline_modified": False,
        "ragtruth_gold_spans_used_as_claims": False,
    }
    assert audit["guardrails"]["GPU_used"] is False
    assert audit["guardrails"]["official_test_opened"] is False
    assert audit["dataset_scope"]["answers"] == 3839
    assert audit["dataset_scope"]["fit_answers"] == 3680
    assert audit["dataset_scope"]["eligible_4bpe_windows"] == 696220
    assert audit["dataset_scope"]["excluded_no_lexical_windows"] == 772

    by_role = {model["role"]: model for model in metadata["models"]}
    extractor = by_role["official_open_claim_triplet_extractor"]
    repc = by_role["official_repc_backbone"]
    nli = by_role["official_nli_checker"]
    localizer = by_role["official_naive_embed_localizer"]
    assert extractor["weight_bytes"] == sum(
        item["bytes"] for item in extractor["weight_files"]
    )
    assert repc["weight_bytes"] == sum(item["bytes"] for item in repc["weight_files"])
    assert extractor["fits_8gib_as_documented"] is False
    assert repc["fits_8gib_as_documented"] is False
    assert nli["fits_8gib_individually_with_small_batch"] is True
    assert localizer["fits_8gib_individually_with_small_batch"] is True
    assert metadata["guardrails"]["weight_files_downloaded"] is False

    files = [
        "REPORT.md",
        "FEASIBILITY_CHECKLIST.json",
        "MODEL_METADATA_SNAPSHOT.json",
        "CPU_AUDIT.json",
        "audit_feasibility_cpu.py",
        "verify_feasibility_cpu.py",
    ]
    result = {
        "schema_version": "refchecker-feasibility-self-check-r32-v1",
        "status": "passed",
        "checks": {
            "official_source_exact_commit": True,
            "official_source_exact_tree": True,
            "official_source_exact_tag": True,
            "official_source_checkout_clean": True,
            "report_has_blocked_verdict": True,
            "json_artifacts_parse": True,
            "weight_sums_match": True,
            "dataset_counts_match": True,
            "guardrails_match": True,
            "formal_score_prohibited": True,
        },
        "official_source": {"commit": commit, "tree": tree, "tag": tag},
        "artifacts_sha256": {name: sha256(HERE / name) for name in files},
    }
    (HERE / "SELF_CHECK.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
