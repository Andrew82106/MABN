"""Rebuild frozen prefix requests and text anchors; not semantic accuracy."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import sys
import types


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite semantic extraction audit")
    manifest = json.loads((run / "RUN.json").read_text(encoding="utf-8"))
    assert manifest["status"] in {"completed", "completed_with_gaps"}, "run is not terminal"
    for relative, expected in manifest["artifact_sha256"].items():
        path = (run / relative).resolve()
        assert path.is_relative_to(run) and sha(path) == expected, f"artifact mismatch: {relative}"
    for relative, expected in manifest["source_snapshot"].items():
        assert sha(run / "source_snapshot" / relative) == expected
    assert sha(run / "runtime_events.jsonl") == manifest["input_sha256"]
    # Isolated frozen package: no accidental import of mutable current code.
    package_name = "paperalpha_semantics_audit_frozen"
    if any(name == package_name or name.startswith(package_name + ".") for name in sys.modules):
        raise RuntimeError("fresh process required for each frozen-source audit")
    package = types.ModuleType(package_name)
    package.__path__ = [str(run / "source_snapshot/code/src/paperalpha_runtime")]
    package.__package__ = package_name
    package.__spec__ = ModuleSpec(package_name, loader=None, is_package=True)
    sys.modules[package_name] = package
    protocol_name = manifest.get("semantic_protocol", "v1")
    assert protocol_name in {"v1", "v2"}, "unsupported frozen semantic protocol"
    module = importlib.import_module(package_name + ".semantic_evidence" + ("_v2" if protocol_name == "v2" else ""))
    spec = json.loads((run / "extraction_specification.json").read_text(encoding="utf-8"))
    assert spec["prompt"] == module.PROMPT and spec["schema"] == module.OUTPUT_SCHEMA
    window = module.TextEvidenceWindow(**spec["window"])
    packets = [packet for row in read_rows(run / "runtime_events.jsonl") if (packet := window.push(row)) is not None]
    selected = packets[:manifest["max_api_calls"]]
    assert len(packets) == manifest["available_text_records"] and len(selected) == manifest["selected_text_records"]
    assert selected == read_rows(run / "input_packets.jsonl"), "prefix input packets do not reconstruct"
    assert window.projector.report() == manifest["journal_report"]
    assert [p["current"]["record_id"] for p in packets[manifest["max_api_calls"]:]] == manifest["unselected_record_ids"]
    def by_call(name):
        values = read_rows(run / name)
        ids = [r["call"] for r in values]
        assert len(ids) == len(set(ids)), "duplicate call result"
        return {r["call"]: r for r in values}
    requests, responses, results = map(by_call, ("api_requests.jsonl", "api_responses.jsonl", "extractions.jsonl"))
    assert set(results) == set(range(1, len(selected) + 1))
    expected_requested = {i for i, p in enumerate(selected, 1) if not p["context"]["current_over_budget"]}
    assert set(requests) == set(responses) == expected_requested
    observations, references = 0, 0
    kinds, relations = Counter(), Counter()
    input_tokens = output_tokens = 0
    for index, packet in enumerate(selected, 1):
        result = results[index]
        assert result["record_id"] == packet["current"]["record_id"]
        assert result["input_packet_sha256"] == module.digest(packet)
        if index not in expected_requested:
            assert result["status"] == "not_requested" and "evidence" not in result
            continue
        request, response = requests[index], responses[index]
        assert request["record_id"] == response["record_id"] == result["record_id"]
        assert request["input_packet_sha256"] == response["input_packet_sha256"] == result["input_packet_sha256"]
        assert request["request"] == module.request_payload(packet), "request differs from frozen protocol"
        if "response" not in response:
            assert response["status"] == result["status"] == "api_error"
            assert response["error_type"] == result["error_type"] and "evidence" not in result
            continue
        usage = response["response"].get("usage") or {}
        assert usage == (result.get("usage") or {})
        input_tokens += usage.get("input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)
        try:
            validated = module.validate_annotation(packet, module.response_annotation(response["response"]))
        except (ValueError, KeyError, TypeError):
            assert result["status"] == "invalid_observation" and "evidence" not in result
            continue
        assert result["status"] == "validated_observation" and result["evidence"] == validated
        # Independent simple slicing/access checks in addition to reexecution.
        available = {d["record_id"]: d for d in packet["prior_records"]}
        for item in validated["observations"]:
            anchor = item["anchor"]
            assert packet["current"]["content"][anchor["start"]:anchor["end"]] == anchor["quote"]
            for part in [*item.get("condition_anchors", []), *([item["quantity_anchor"]] if item.get("quantity_anchor") else [])]:
                assert anchor["start"] <= part["start"] < part["end"] <= anchor["end"]
                assert packet["current"]["content"][part["start"]:part["end"]] == part["quote"]
            observations += 1
            kinds[item["kind"]] += 1
            for relation in item["references"]:
                anchor = relation["anchor"]
                previous = available[anchor["record_id"]]
                assert previous["sequence"] < packet["current"]["sequence"]
                assert previous["actor"] == packet["current"]["actor"] or previous["target"] == packet["current"]["actor"]
                assert previous["content"][anchor["start"]:anchor["end"]] == anchor["quote"]
                references += 1
                relations[relation["relation_hypothesis"]] += 1
    assert input_tokens == manifest["input_tokens"] and output_tokens == manifest["output_tokens"]
    assert observations == manifest["validated_observations"] and references == manifest["validated_reference_anchors"]
    counts = dict(Counter(r["status"] for r in results.values()))
    assert counts == manifest["outcome_counts"]
    expected_status = "completed" if set(counts) == {"validated_observation"} else "completed_with_gaps"
    assert manifest["status"] == expected_status
    report = {"status": "passed", "run": str(run), "run_status": manifest["status"],
        "checked_artifacts": len(manifest["artifact_sha256"]), "reconstructed_prefix_packets": len(selected),
        "api_requests": len(requests), "outcome_counts": counts, "observation_anchors": observations,
        "prior_reference_anchors": references, "observation_kinds": dict(kinds), "relation_hypotheses": dict(relations),
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "scope": "Frozen request/protocol/output integrity and quote anchoring. No independent semantic classification, provenance or safety accuracy."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.out), ensure_ascii=False))
