"""Verify a frozen balanced API cohort without making network calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
sys.path.insert(0, str(ROOT / 'code/commands'))
from harnessaudit_cohort import sha, write_new
from collect_balanced_delegation_api_v1 import load
from paperalpha_runtime.balanced_delegation_pilot import monitor_configuration


FORBIDDEN = ("reference_label", "reference_policy_violation", "authorised_resource",
             "reference_label_basis", "revocation_before_worker")


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _contains_forbidden(value):
    if isinstance(value, dict):
        return any(k in FORBIDDEN or _contains_forbidden(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden(v) for v in value)
    return False


def verify(run: Path):
    plan = load(run / "PLAN.json")
    report = json.loads((run / "REPORT.json").read_text(encoding="utf-8"))
    assert report["plan_sha256"] == plan["sha256"]
    rows, checked_files = [], 0
    for case in plan["cases"]:
        directory = run / "runs" / case["id"]
        manifest = json.loads((directory / "RUN.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        assert manifest["plan_sha256"] == plan["sha256"]
        assert json.loads((directory / "monitor_config.json").read_text(encoding="utf-8")) == monitor_configuration(case)
        assert manifest["reference_label_expected"] == case["reference_label"]
        assert manifest["violations"] == int(manifest["episode_policy_violation"])
        assert manifest["violations"] == int(case["reference_label"])
        requests, responses = _lines(directory / "api_requests.jsonl"), _lines(directory / "api_responses.jsonl")
        assert len(requests) == manifest["API_calls"] == len(responses)
        assert all(r["model_requested"] == plan["settings"]["model"] for r in requests)
        monitor_events = _lines(directory / "monitor_events.jsonl")
        predictions = _lines(directory / "online_predictions.jsonl")
        assert not _contains_forbidden(json.loads((directory / "monitor_config.json").read_text()))
        assert not _contains_forbidden(monitor_events)
        assert not _contains_forbidden(predictions)
        artifacts = json.loads((directory / "ARTIFACTS.json").read_text())
        for name, expected in artifacts.items():
            assert sha(directory / name) == expected
            checked_files += 1
        rows.append({"case_id": case["id"], "variant": case["variant"],
                     "label": bool(case["reference_label"]), "violations": manifest["violations"],
                     "API_calls": manifest["API_calls"], "monitor_events": len(monitor_events)})
    assert len(rows) == len(plan["cases"])
    result = {"plan_sha256": plan["sha256"], "cases": rows,
              "completed_cases": len(rows), "expected_positives": sum(r["label"] for r in rows),
              "observed_violations": sum(r["violations"] for r in rows),
              "monitor_label_leakage": False, "network_calls": 0,
              "artifacts_checked": checked_files,
              "claim": "Acquisition integrity and evaluator/monitor boundary only; no external benchmark claim."}
    write_new(run / "VERIFICATION.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run.resolve())
