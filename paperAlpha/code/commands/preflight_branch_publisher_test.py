"""Verify real frozen scoring subprocess on development inputs, before test API."""
import argparse
import json
from pathlib import Path
import sys


def run(folder):
    sys.path[:0] = [str(folder / "source_snapshot/code/commands"), str(folder / "source_snapshot/code/src")]
    from collect_branch_publisher_test import load, Monitor, sha, write
    plan = load(folder)
    fit = Path(plan["source_fit"])
    frozen = json.loads((folder / "bundle/FROZEN.json").read_text(encoding="utf-8"))
    candidates = {family: json.loads((fit / "candidates" / (identity + ".json")).read_text(encoding="utf-8"))
                  for family, identity in frozen["selected"].items()}
    checked, maximum = 0, 0.
    monitor = Monitor(folder, plan["bn_python"])
    try:
        for split in ("train", "validation"):
            rows = json.loads((fit / "inputs" / (split + ".json")).read_text(encoding="utf-8"))
            for i, row in enumerate(rows):
                prefix = row["prefix"]
                result = monitor.score({"observation": prefix["observation"], "prefix_sequence": prefix["seq"]})
                assert result["selected"] == frozen["selected"]
                for family, actual in result["count_distributions"].items():
                    expected = candidates[family]["distributions"][split][i]
                    maximum = max(maximum, max(abs(a - b) for a, b in zip(actual, expected)))
                    assert len(actual) == len(expected) == 3
                    checked += 1
    finally:
        monitor.close()
    assert checked == 128 and maximum < 1e-14
    assert not (folder / "test").exists()
    result = {"plan_sha256": plan["sha256"], "bundle_sha256": plan["bundle_sha256"], "source_sha256": sha(Path(__file__)),
        "development_model_predictions_checked": checked, "max_difference": maximum,
        "used_actual_isolated_monitor_process": True, "test_api_calls": 0,
        "scope": "Read-only numerical qualification on prior development inputs; no model changes or test responses"}
    write(folder / "PREFLIGHT.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run(args.run.resolve())
