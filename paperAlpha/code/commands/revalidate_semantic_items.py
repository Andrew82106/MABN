"""Derive item decisions from saved v2 responses without new API calls."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_runtime.semantic_observation_batch import validate_items
from audit_semantic_extraction import audit, read_rows, sha


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def derive(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("item derivation must use a new output directory")
    manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
    if manifest.get("semantic_protocol") != "v2":
        raise ValueError("only typed v2 responses can use this derivation")
    output.mkdir(parents=True)
    source_audit = audit(run, output / "SOURCE_AUDIT.json")
    frozen = importlib.import_module("paperalpha_semantics_audit_frozen.semantic_evidence_v2")
    packets = read_rows(run / "input_packets.jsonl")
    responses = {r["call"]: r for r in read_rows(run / "api_responses.jsonl")}
    original = {r["call"]: r for r in read_rows(run / "extractions.jsonl")}
    batches, predicates, relations, errors = [], Counter(), Counter(), Counter()
    for call, packet in enumerate(packets, 1):
        result = {"call": call, "record_id": packet["current"]["record_id"], "original_status": original[call]["status"]}
        response = responses.get(call, {})
        if "response" not in response:
            result.update(status="unavailable", reason=original[call]["status"])
        else:
            try:
                proposed = frozen.response_annotation(response["response"])
                batch = validate_items(packet, proposed, strict_validator=frozen.validate_annotation)
            except (ValueError, KeyError, TypeError) as exc:
                result.update(status="unavailable", reason="invalid_envelope_or_response", detail=str(exc))
            else:
                result.update(status="item_decisions", batch=batch)
                for item in batch["items"]:
                    if item["status"] == "rejected":
                        errors[item["reason"] + ": " + item["detail"]] += 1
                        continue
                    obs = item["observation"]
                    predicates[obs["predicate"]] += 1
                    relations.update(ref["relation_hypothesis"] for ref in obs["references"])
        batches.append(result)
    save(output / "batches.json", batches)
    decisions = [item for r in batches for item in r.get("batch", {}).get("items", [])]
    sources = {}
    for path in [Path(__file__), ROOT / 'code/commands/audit_semantic_extraction.py',
                 ROOT / 'code/src/paperalpha_runtime/semantic_observation_batch.py',
                 ROOT / 'code/src/paperalpha_runtime/semantic_evidence.py',
                 ROOT / 'code/src/paperalpha_runtime/semantic_evidence_v2.py']:
        destination = output / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        sources[path.relative_to(ROOT).as_posix()] = sha(destination)
    report = {"status": "completed", "source_run": str(run), "source_manifest_sha256": sha(run / "RUN.json"),
        "source_artifact_sha256": manifest["artifact_sha256"], "source_audit_status": source_audit["status"],
        "new_api_calls": 0, "source_snapshot": sources, "records": len(batches),
        "record_status_counts": dict(Counter(r["status"] for r in batches)),
        "acceptance_status_counts": dict(Counter(r["batch"]["acceptance_status"] for r in batches if "batch" in r)),
        "proposed_items": len(decisions), "accepted_items": sum(i["status"] == "accepted" for i in decisions),
        "rejected_items": sum(i["status"] == "rejected" for i in decisions),
        "retained_from_originally_rejected_records": sum(r.get("batch", {}).get("accepted_items", 0)
            for r in batches if r["original_status"] != "validated_observation"),
        "accepted_predicates": dict(predicates), "relation_hypotheses": dict(relations), "rejection_reasons": dict(errors),
        "scope": "Post-hoc item-level structural derivation. Original whole-record protocol/results unchanged; not semantic accuracy or BN input truth."}
    report["artifact_sha256"] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    save(output / "REPORT.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = derive(args.run, args.out)
    print(json.dumps({k: report[k] for k in ("status", "records", "acceptance_status_counts", "accepted_items", "rejected_items", "new_api_calls")}, ensure_ascii=False))
