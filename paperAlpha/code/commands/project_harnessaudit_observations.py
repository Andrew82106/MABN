"""Split native HarnessAudit JSONL into model observations and offline labels.

Does not train a model, invent live hooks, or treat test fixtures as real runs.
Use the required provenance argument to distinguish data acquisition methods.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.harnessaudit_observations import SCHEMA, SOURCE_REVISION, canonical, project_records


def project_file(input_path, output_dir, *, provenance):
    source, output = Path(input_path).resolve(), Path(output_dir).resolve()
    if provenance not in {"test_fixture", "native_execution", "external_unverified"}:
        raise ValueError("explicit supported provenance is required")
    if output.exists():
        raise ValueError("output must be a new directory; existing artifacts are never overwritten")
    raw = source.read_bytes()
    rows = []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"invalid JSON at line {line_number}") from exc
    bundles = project_records(rows)
    report = {"schema_version": SCHEMA, "source": str(source),
              "source_sha256": hashlib.sha256(raw).hexdigest(), "source_bytes": len(raw),
              "provenance_declared_by_caller": provenance, "provenance_independently_verified": False,
              "source_semantics_reference_commit": SOURCE_REVISION,
              "traces": len(bundles), "observations": sum(len(b["monitor"]["events"]) for b in bundles),
              "eligible_rule_violation_labels": sum(b["evaluation"]["eligible"] for b in bundles),
              "online_warning_eligible": False, "direct_bn_evaluation_eligible": False,
              "scope": "Observation contract only; no training, runtime collection, safety accuracy or MAS baseline comparison.",
              "input_rule": "Only the monitor payload is model input. Join IDs and all other files are excluded.",
              "trace_qualification": [{"trace_id": b["trace_id"], "monitor_sha256": b["monitor_sha256"],
                                       "ineligibility_reasons": b["evaluation"]["ineligibility_reasons"],
                                       "model_readiness": b["model_readiness"]} for b in bundles]}
    output.mkdir(parents=True, exist_ok=False)
    products = {
        "monitor_inputs.jsonl": [{"trace_id": b["trace_id"], "monitor": b["monitor"]} for b in bundles],
        "evaluation_only.jsonl": [{"trace_id": b["trace_id"], **b["evaluation"]} for b in bundles],
        "metadata_only.jsonl": [{"trace_id": b["trace_id"], **b["metadata"]} for b in bundles],
    }
    for name, records in products.items():
        (output / name).write_text("".join(canonical(r) + "\n" for r in records), encoding="utf-8")
    report["artifacts_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in products}
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    report["source_snapshot_sha256"] = {}
    for path in (Path(__file__), ROOT / 'code/src/paperalpha_monitor/harnessaudit_observations.py'):
        source_bytes = path.read_bytes()
        (snapshot / path.name).write_bytes(source_bytes)
        report["source_snapshot_sha256"][path.name] = hashlib.sha256(source_bytes).hexdigest()
    (output / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", required=True, choices=("test_fixture", "native_execution", "external_unverified"))
    args = parser.parse_args()
    report = project_file(args.input, args.out, provenance=args.provenance)
    print(json.dumps({key: report[key] for key in ("traces", "observations", "eligible_rule_violation_labels", "scope")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
