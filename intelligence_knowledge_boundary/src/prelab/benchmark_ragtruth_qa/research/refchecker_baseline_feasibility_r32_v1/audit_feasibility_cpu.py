"""Static/CPU feasibility audit for the unmodified RefChecker baseline.

Only the frozen QA fit and calibration partitions are read. No model weights,
GPU, network inference, or sealed official test files are accessed.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / "third_party/reference/refchecker_1df1b25"

DATA = {
    "fit_rows": ROOT / "fit_expansion/data/fit.jsonl",
    "cal_rows": ROOT / "data/calibration.jsonl",
    "fit_answers": ROOT / "fit_expansion/data/answers_fit.jsonl",
    "cal_answers": ROOT / "data/answers_calibration.jsonl",
    "fit_windows": ROOT / "fit_expansion/data/windows_k4_fit.jsonl",
    "cal_windows": ROOT / "data/windows_k4_calibration.jsonl",
    "fit_excluded_windows": ROOT / "fit_expansion/data/windows_excluded_fit.jsonl",
    "cal_excluded_windows": ROOT / "data/windows_excluded_calibration.jsonl",
    "gold_manifest": ROOT / "data/gold_manifest.json",
}
SOURCE = {
    "extractor": REF / "refchecker/extractor/llm_extractor.py",
    "extractor_parser": REF / "refchecker/extractor/extractor_base.py",
    "nli": REF / "refchecker/checker/nli_checker.py",
    "checker_base": REF / "refchecker/checker/checker_base.py",
    "repc": REF / "refchecker/checker/repc/repc_checker.py",
    "localizer": REF / "refchecker/localizer/embed_localizer.py",
    "aggregator": REF / "refchecker/aggregator.py",
    "benchmark_driver": REF / "benchmark/evaluation/autocheck.py",
    "package": REF / "pyproject.toml",
    "license": REF / "LICENSE",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def nonempty_line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def dist(values: list[int]) -> dict:
    values = sorted(values)
    n = len(values)
    return {
        "min": values[0],
        "mean": sum(values) / n,
        "median": statistics.median(values),
        "p90": values[math.ceil(0.90 * n) - 1],
        "p95": values[math.ceil(0.95 * n) - 1],
        "max": values[-1],
    }


PASSAGE_HEADER = re.compile(r"(?im)^passage\s+(\d+)\s*:")
TOKENISH = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def passage_bodies(text: str) -> list[tuple[int, str]]:
    matches = list(PASSAGE_HEADER.finditer(text))
    out = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((int(match.group(1)), text[match.end():end].strip()))
    return out


def main() -> None:
    assert REF.is_dir()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REF, text=True
    ).strip()
    tag = subprocess.check_output(
        ["git", "describe", "--tags", "--exact-match", "HEAD"], cwd=REF, text=True
    ).strip()
    assert commit == "1df1b25cee792ba2b171302e31ca4f768bd67703"
    assert tag == "v0.2.17"
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REF, text=True
    ).strip()

    rows = read_jsonl(DATA["fit_rows"]) + read_jsonl(DATA["cal_rows"])
    answers = read_jsonl(DATA["fit_answers"]) + read_jsonl(DATA["cal_answers"])
    gold = json.loads(DATA["gold_manifest"].read_text(encoding="utf-8"))
    assert len(rows) == len(answers) == 3839
    assert [r["response_id"] for r in rows] == [a["response_id"] for a in answers]
    assert all(r["partition"] == "fit" for r in rows[:3680])
    assert all(r["partition"] == "calibration" for r in rows[3680:])
    fit_groups = {r["group_id"] for r in rows[:3680]}
    cal_groups = {r["group_id"] for r in rows[3680:]}
    assert len(fit_groups) == 615 and len(cal_groups) == 154
    assert fit_groups.isdisjoint(cal_groups)
    assert gold["official_test_opened"] is False

    eligible_fit = nonempty_line_count(DATA["fit_windows"])
    eligible_cal = nonempty_line_count(DATA["cal_windows"])
    excluded_fit = nonempty_line_count(DATA["fit_excluded_windows"])
    excluded_cal = nonempty_line_count(DATA["cal_excluded_windows"])
    assert (eligible_fit, eligible_cal) == (653979, 42241)
    assert (excluded_fit, excluded_cal) == (692, 80)

    passage_counts, passage_tokenish, answer_token_counts = [], [], []
    segments_lower = []
    malformed = []
    for row, answer in zip(rows, answers):
        passages = passage_bodies(row["retrieved_passages"])
        ids = [x[0] for x in passages]
        if ids != [1, 2, 3]:
            malformed.append({"response_id": row["response_id"], "passage_ids": ids})
        counts = [len(TOKENISH.findall(body)) for _, body in passages]
        passage_counts.append(len(passages))
        passage_tokenish.extend(counts)
        # A lower-bound approximation to RefChecker split_text(., 200).
        segments_lower.append(sum(max(1, math.ceil(n / 200)) for n in counts))
        answer_token_counts.append(int(answer["token_count"]))

    assert not malformed
    paper_noisy_claims_per_answer = 4.9
    paper_noisy_response_tokens = 61.3
    projected_claims_same_rate = len(rows) * paper_noisy_claims_per_answer
    projected_claims_length_scaled = projected_claims_same_rate * (
        statistics.mean(answer_token_counts) / paper_noisy_response_tokens
    )
    avg_segments = statistics.mean(segments_lower)

    nli_text = SOURCE["nli"].read_text(encoding="utf-8")
    base_text = SOURCE["checker_base"].read_text(encoding="utf-8")
    repc_text = SOURCE["repc"].read_text(encoding="utf-8")
    loc_text = SOURCE["localizer"].read_text(encoding="utf-8")
    driver_text = SOURCE["benchmark_driver"].read_text(encoding="utf-8")
    assert "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli" in nli_text
    assert 'device_map="cuda:1"' in repc_text
    assert "teknium/OpenHermes-2.5-Mistral-7B" in repc_text
    assert "princeton-nlp/sup-simcse-roberta-large" in loc_text
    assert "max_reference_segment_length = 200" in driver_text
    assert "merge_psg=True" in driver_text
    assert "is_joint: bool = True" in base_text

    out = {
        "status": "cpu_static_audit_complete_no_baseline_inference",
        "official_source": {
            "repository": "https://github.com/amazon-science/RefChecker",
            "commit": commit,
            "tag": tag,
            "clean_checkout": True,
            "license": "Apache-2.0",
            "files_sha256": {k: sha256(v) for k, v in SOURCE.items()},
        },
        "dataset_scope": {
            "partitions_read": ["fit", "calibration"],
            "answers": len(rows),
            "fit_answers": 3680,
            "calibration_answers": 159,
            "fit_material_groups": len(fit_groups),
            "calibration_material_groups": len(cal_groups),
            "fit_cal_material_group_overlap": len(fit_groups & cal_groups),
            "eligible_4bpe_windows": eligible_fit + eligible_cal,
            "eligible_4bpe_windows_by_partition": {
                "fit": eligible_fit,
                "calibration": eligible_cal,
            },
            "excluded_no_lexical_windows": excluded_fit + excluded_cal,
            "excluded_no_lexical_windows_by_partition": {
                "fit": excluded_fit,
                "calibration": excluded_cal,
            },
            "passages": sum(passage_counts),
            "passages_per_answer": dist(passage_counts),
            "response_raw_bpe_tokens": dist(answer_token_counts),
            "passage_tokenish_count": dist(passage_tokenish),
            "approx_reference_segments_200_per_answer": dist(segments_lower),
            "approx_reference_segments_total": sum(segments_lower),
            "note": "Tokenish/segment counts are CPU planning estimates; exact RefChecker split_text requires spaCy en_core_web_sm.",
        },
        "pair_projection": {
            "paper_noisy_context_claims_per_response": paper_noisy_claims_per_answer,
            "paper_noisy_context_response_tokens": paper_noisy_response_tokens,
            "same_claim_rate_claims": projected_claims_same_rate,
            "length_scaled_claims_rough": projected_claims_length_scaled,
            "same_rate_checker_pairs": projected_claims_same_rate * avg_segments,
            "length_scaled_checker_pairs_rough": projected_claims_length_scaled * avg_segments,
            "localizer_forward_lower_same_rate": projected_claims_same_rate * 4,
            "localizer_forward_lower_length_scaled": projected_claims_length_scaled * 4,
            "caveat": "Actual claims are unknown until the fixed extractor runs; scaling claims linearly with response length is only a capacity bound, not a result.",
        },
        "static_interface_findings": {
            "native_output": "One hard Entailment/Neutral/Contradiction label per extracted claim-triplet; optional strict/soft/major response aggregation.",
            "no_native_answer_offsets": True,
            "nli_requires_is_joint_false": True,
            "noisy_context_segment_length": 200,
            "noisy_context_merge_passages": True,
            "repc_hardcoded_backbone_device": "cuda:1",
            "repc_public_check_dispatcher_runnable_as_is": False,
            "localizer_output": "HTML-decorated token text, not numeric spans",
            "localizer_default_thresholds": [0.65, 0.6, 0.65],
        },
        "guardrails": {
            "GPU_used": False,
            "weights_downloaded": False,
            "official_test_opened": False,
            "gold_spans_used_as_refchecker_claim_input": False,
            "formal_baseline_modified": False,
        },
    }
    (HERE / "CPU_AUDIT.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "answers": len(rows),
        "passages": sum(passage_counts),
        "avg_response_bpe": statistics.mean(answer_token_counts),
        "avg_segments": avg_segments,
        "claim_projection": [projected_claims_same_rate, projected_claims_length_scaled],
        "pair_projection": [projected_claims_same_rate * avg_segments, projected_claims_length_scaled * avg_segments],
    }, indent=2))


if __name__ == "__main__":
    main()
