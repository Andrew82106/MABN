"""Freeze and inspect branch-preserving development inputs and simple rules."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/src'), str(ROOT / 'code/commands')]
from collect_privacy_transfer_api import sha, write
from paperalpha_monitor.branch_privacy_input import BranchPrivacyInput
from paperalpha_monitor.scoped_privacy_monitor import gated_distribution


def prepare(data_run, audited, numerical_run, out):
    if out.exists():
        raise ValueError("new compiled-input directory required")
    snapshot = numerical_run / "source_snapshot"
    files = {p.relative_to(snapshot).as_posix(): p for p in (snapshot / 'code/src').rglob("*.py")}
    files.update({s: ROOT / s for s in ('code/commands/replay_branch_privacy_inputs.py', 'code/commands/collect_privacy_transfer_api.py',
        'code/src/paperalpha_monitor/branch_privacy_input.py', 'code/tests/test_branch_privacy_input.py')})
    # The acquisition runtime/prompt definitions are from their earlier freeze.
    for p in (data_run / "source_snapshot/code/src/paperalpha_runtime").glob("*.py"):
        files['code/src/paperalpha_runtime/' + p.name] = p
    for name, source in files.items():
        dest = out / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    write(out / "PLAN.json", {"data_run": str(data_run), "audited": str(audited),
        "input_sha256": {name: sha(audited / name) for name in ("REPORT.json", "train.json", "validation.json")},
        "source_sha256": {name: sha(out / "source_snapshot" / name) for name in files},
        "scope": "Post-acquisition input compilation and predeclared-rule development diagnosis; not fitted monitoring or held-out performance"})
    print(json.dumps({"prepared": str(out), "source_files": len(files)}), flush=True)


def score_prefix(prefix):
    sample = BranchPrivacyInput(prefix["observation"], available_journal_sequence=prefix["seq"], prefix_journal_sequence=prefix["seq"])
    sample.validate()
    presence = sample.worker_presence.any(axis=1).astype(float)
    support = sample.release.visible_features[:, 0]
    scores = {}
    for method, q in (("persistence_shared_gate", presence), ("visible_compliance_shared_gate", presence * support)):
        dist = gated_distribution(sample.release, q, .5)
        scores[method] = {str(k): float(dist[k:].sum()) for k in (1, 2)}
    return sample, scores


def run(out):
    if Path(__file__).resolve() != out / "source_snapshot/code/commands/replay_branch_privacy_inputs.py":
        raise ValueError("run frozen inspector")
    import paperalpha_monitor.branch_privacy_input as implementation
    assert Path(implementation.__file__).resolve() == out / "source_snapshot/code/src/paperalpha_monitor/branch_privacy_input.py"
    plan = json.loads((out / "PLAN.json").read_text(encoding="utf-8"))
    for name, expected in plan["source_sha256"].items():
        assert sha(out / "source_snapshot" / name) == expected
    parent = Path(plan["audited"])
    for name, expected in plan["input_sha256"].items():
        assert sha(parent / name) == expected
    rows = []
    for split in ("train", "validation"):
        for row in json.loads((parent / (split + ".json")).read_text(encoding="utf-8")):
            assert row["split"] == split
            sample, predictions = score_prefix(row["prefix"])
            rows.append({"id": row["id"], "split": split, "specification": sample.specification(), "predictions": predictions,
                "targets": row["outcome"]["targets"]})
    summary = {}
    for split in ("train", "validation"):
        subset = [r for r in rows if r["split"] == split]
        summary[split] = {}
        for method in ("persistence_shared_gate", "visible_compliance_shared_gate"):
            summary[split][method] = {str(k): {"n": len(subset), "positives": sum(r["targets"][str(k)] for r in subset),
                "brier": float(np.mean([(r["predictions"][method][str(k)] - r["targets"][str(k)]) ** 2 for r in subset]))} for k in (1, 2)}
    write(out / "compiled_inputs.json", rows)
    write(out / "REPORT.json", {"compiled_development_prefixes": len(rows), "worker_matrices": [r["specification"]["worker_presence"] for r in rows],
        "all_legacy_pool_agree_on_this_cohort": all(r["specification"]["legacy_serialized_pool_agrees"] for r in rows),
        "rule_diagnostics": summary, "test_outputs_used": False, "model_fitted": False, "new_api_calls": 0,
        "source_files_verified": len(plan["source_sha256"]), "scope": plan["scope"]})
    print(json.dumps({"compiled_prefixes": len(rows), "rule_diagnostics": summary, "test_used": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--data-run", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--numerical-run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.data_run.resolve(), args.audit.resolve(), args.numerical_run.resolve(), args.out.resolve())
    else:
        run(args.out.resolve())
