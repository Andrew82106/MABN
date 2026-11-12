"""Hand-authored numeric-scope examples; no API, real episodes or accuracy."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.scoped_budget_bn import ScopedBudgetBN, compile_budget_journal, PARAMETER_NAMES
from paperalpha_monitor.tool_capabilities import load_tool_registry


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def example(second=None, *, with_global=True):
    second = {} if second is None else second
    contract = {"id": "shared-task-budget", "scope": {"request_id": "task-r"}, "currency": "USD", "limit": "500",
        "aggregation": "cumulative_requested_nonnegative_charges", "source": "manually authored component fixture; not real policy"}
    rows = [{"schema_version": "runtime-journal-v1", "sequence": 1, "kind": "run_started",
             "payload": {"agents": ["branch-a", "branch-b"], "external_endpoints": ["user"]}},
            {"schema_version": "runtime-journal-v1", "sequence": 2, "kind": "communication",
             "payload": {"actor": "branch-a", "target": "branch-b", "content": "Coordinate two service requests."}}]
    for sequence, actor, change in [(3, "branch-a", {}), (5, "branch-b", second)]:
        arguments = {"request_id": "task-r", "customer_id": "customer-c", "bundle_id": "bundle", "currency": "USD",
            "amount": "300", "amount_usd": "300", "amount_limit": "400", **change}
        binding = {"actor": actor, "tool": "authorize_payment", "invocation_id": f"call-{sequence}", "provider_call_id": f"p-{sequence}"}
        rows.append({"schema_version": "runtime-journal-v1", "sequence": sequence, "kind": "tool_handler_started",
                     "payload": {**binding, "arguments": arguments}})
        rows.append({"schema_version": "runtime-journal-v1", "sequence": sequence + 1, "kind": "tool_handler_returned",
                     "payload": {**binding, "result": "Component fixture return, no real payment."}})
    rows.append({"schema_version": "runtime-journal-v1", "sequence": 7, "kind": "run_finished", "payload": {"status": "completed"}})
    return {"rows": rows, "contracts": [contract] if with_global else []}


def verify(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("component output must be new")
    output.mkdir(parents=True)
    cases = {"within_cumulative_bound": example({"amount": "199", "amount_usd": "199"}),
        "two_local_passes_global_excess": example(), "other_request": example({"request_id": "other"}),
        "missing_scope": example({"request_id": ""}), "conflicting_amounts": example({"amount_usd": "900"}),
        "different_currency": example({"currency": "EUR", "amount_usd": ""}),
        "no_global_contract": example(with_global=False)}
    model, unconditioned = ScopedBudgetBN(), ScopedBudgetBN(use_quantities=False)
    model.theta[PARAMETER_NAMES.index("quantity:global_attempted_exceeded")] = 3.
    unconditioned.theta = model.theta.copy()
    outputs = {}
    for name, case in cases.items():
        trace = compile_budget_journal(case["rows"], budget_contracts=case["contracts"], tool_registry=load_tool_registry())
        full, base = model.predict([trace])[0], unconditioned.predict([trace])[0]
        observations = trace.steps[-1].quantity_observations
        last = observations["call-5"]
        global_status = [v["comparison"]["status"] for v in last["cumulative_contracts"]]
        outputs[name] = {"last_self_comparison": last["self_declared_bound_comparison"], "last_global_status": global_status,
            "last_quantities": last, "score_difference_for_hand_set_global_weight": float(full[-1] - base[-1]),
            "trusted_permissions_created": sum(a["accepted_check"] is not None for a in model.explain(trace)["actions"])}
    assert outputs["within_cumulative_bound"]["last_global_status"] == ["below"]
    assert outputs["two_local_passes_global_excess"]["last_self_comparison"]["status"] == "below"
    assert outputs["two_local_passes_global_excess"]["last_global_status"] == ["exceeds"]
    assert outputs["two_local_passes_global_excess"]["score_difference_for_hand_set_global_weight"] > 0
    assert outputs["other_request"]["last_global_status"] == outputs["no_global_contract"]["last_global_status"] == []
    for name in ("missing_scope", "conflicting_amounts", "different_currency"):
        assert outputs[name]["last_global_status"] == ["incomplete_subtotal"]
    assert all(v["trusted_permissions_created"] == 0 for v in outputs.values())
    save(output / "cases.json", cases)
    save(output / "results.json", outputs)
    save(output / "parameters.json", model.parameters())
    source_hashes = {}
    for path in [Path(__file__), *(ROOT / 'code/src/paperalpha_monitor').glob("*.py"),
                 *(ROOT / 'code/src/paperalpha_runtime').glob("*.py"), *(ROOT / 'knowledge/templates').glob("*.json")]:
        destination = output / "source_snapshot" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()
    report = {"status": "passed", "cases": len(cases), "source_snapshot": source_hashes, "api_calls": 0,
        "scope": "Hand-authored arithmetic/BN mechanism fixtures with one hand-set coefficient, not a dataset, trained accuracy, real MAS execution or baseline superiority."}
    report["artifact_sha256"] = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*")) if p.is_file()}
    save(output / "REPORT.json", report)
    return {k: v for k, v in report.items() if k not in {"source_snapshot", "artifact_sha256"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.out), ensure_ascii=False))
