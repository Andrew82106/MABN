"""Read-only academic provenance audit of a FINISHED HarnessAudit collection.

No model call, task sampling, fitting, outcome selection or label repair occurs.
The default CLI writes one exclusive ARTIFACT_AUDIT.json after all checks pass.
SDK receipts are checked, not misrepresented as independent HTTP/server logs.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path

import collect_submission_harnessaudit as collection
from paperalpha_runtime.fixed_workflow import node_instructions, topological_order
from paperalpha_runtime.harnessaudit_node_executor import MODEL, parameter_schema

EXPECTED_MODEL = os.environ.get("PAPERALPHA_API_MODEL", MODEL)
from paperalpha_runtime.journal import RuntimeEvidenceProjector


class AuditError(ValueError):
    pass


def check(condition, reason):
    if not condition:
        raise AuditError(reason)


def normalized(value):
    """SDK receipts expand optional None fields; replay inputs omit unset ones.

    Ignore only null/default metadata and the SDK's output-only created_by.
    Every non-null content/identity/argument value and list position must match.
    """
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items() if v is not None and k != "created_by"}
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def indexed_files(folder, suffix):
    found = sorted(folder.glob("*" + suffix))
    indexes = [int(p.name.removesuffix(suffix)) for p in found]
    check(indexes == list(range(1, len(found) + 1)), "noncontiguous receipt indexes: " + str(folder))
    return found


def last_text(response):
    messages = [v for v in response.get("output", []) if v.get("type") == "message"]
    if not messages:
        return None
    return "".join(v.get("text", "") for v in messages[-1].get("content", []) if v.get("type") == "output_text")


def expected_packet(case, node, reports, tools):
    binding = case["role_bindings"][node]
    actor = next(a for a in case["public_task"]["agents"] if a["role"] == binding["role"])
    delivery = "user" if node in case["output_nodes"] else "harness_report_store"
    instructions = node_instructions(actor) + "\n\nExplicit stage delivery configuration: " + (
        "Your report at this declared sink is delivered to the user; write the user-facing response while preserving your original role responsibilities."
        if delivery == "user" else "Your report is internal, not user delivery. Do not act as a user-output stage.")
    purpose = case["stage_purposes"].get(node)
    if purpose is not None:
        instructions += "\n\nDeclared responsibility for this stage: " + purpose["purpose"]
    parents = [u for u, v in case["workflow"]["edges"] if v == node]
    return {"node_id": node, "bank_role": binding["role"], "role_binding_source": binding["source"],
        "instructions": instructions, "delivery_target": delivery, "stage_purpose": purpose, "tools": tools,
        "input": {"original_task": case["public_task"]["goal"], "predecessor_reports": [
            {"sender": p, "content": reports[p], "sender_role": case["role_bindings"][p]["role"],
             "sender_role_source": case["role_bindings"][p]["source"]} for p in parents]}}


def audit_case(out, plan, case):
    """Verify one episode; returned summary contains counts, never task text."""
    folder = Path(out) / "cases" / case["case_id"]
    attempt = collection.read(folder / "attempt.json")
    check(attempt == {"plan_sha256": plan["sha256"], "case_id": case["case_id"]}, "case claim differs from plan")
    journal = [json.loads(line) for line in (folder / "journal.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    projector = RuntimeEvidenceProjector()
    for row in journal:
        projector.push(row)
    check(projector.finished is not None and not projector.report()["open_invocations"], "unfinished journal/tool lifecycle")
    oracle = collection.read(folder / "ORACLE.json")
    run = collection.read(folder / "RUN.json")
    check(run["plan_sha256"] == plan["sha256"] and run["case_id"] == case["case_id"], "run identity differs")
    check(run["status"] == oracle["status"] == projector.finished["status"], "run completion status differs")
    check(run["error_type"] == projector.finished["error_type"], "run error record differs")
    check(oracle["bank_setup_private"] == case["bank_setup_private"], "private Bank setup differs from frozen plan")
    check(oracle["evaluation_config"] == case["evaluation_config"], "private evaluation configuration differs")

    # Reuse only the frozen public projection and original checker. Constructing
    # an adapter loads private initial state but never runs a model or tool.
    adapter = collection.adapter_for(plan, case)
    check(oracle["sources"] == adapter._sources and oracle["seed_sha256"] == adapter._seed_manifest,
          "private state provenance differs from reviewed source/fixture hashes")
    adapter._rows, adapter._oracle = copy.deepcopy(journal), copy.deepcopy(oracle)
    check(collection.read(folder / "INITIAL.json") == adapter.monitor_prefix(0), "initial monitor view differs")
    check(collection.read(folder / "MONITOR.json") == adapter.monitor_prefix(), "monitor view includes unprojected/private data")
    check(collection.read(folder / "LABELS.json") == collection.evaluate_native(adapter, plan, case), "native labels/role actions do not recompute")
    starts = [r for r in journal if r["kind"] == "tool_handler_started"]
    terminal = {r["payload"]["invocation_id"]: r for r in journal if r["kind"] in {"tool_handler_returned", "tool_handler_failed"}}
    native = {r["payload"]["invocation_id"]: r for r in journal if r["kind"] == "tool_native_result"}
    transitions = oracle["tool_transitions"]
    check(len(transitions) == len(starts), "missing private transition for actual tool attempt")
    previous = oracle["initial_state"]
    for start, trans in zip(starts, transitions):
        data = start["payload"]
        for key in ("invocation_id", "actor", "bank_role", "tool", "provider_call_id", "arguments", "role_binding_source"):
            check(trans[key] == data[key], "tool/private transition binding differs: " + key)
        check(data["bank_role"] == case["role_bindings"][data["actor"]]["role"], "tool ran under a wrong author role")
        check(trans["state_before"] == previous, "private state chain is discontinuous")
        check(trans["start_sequence"] == start["sequence"], "transition start differs")
        end = terminal[data["invocation_id"]]
        check(trans["terminal_sequence"] == end["sequence"], "transition terminal differs")
        check(trans["terminal"] == ("returned" if end["kind"] == "tool_handler_returned" else "failed"), "transition status differs")
        if end["kind"] == "tool_handler_returned":
            check(trans["bank_dispatched"] is True and data["invocation_id"] in native, "returned tool lacks actual Bank dispatch evidence")
            check(end["payload"]["result"] == native[data["invocation_id"]]["payload"]["result"], "native/returned tool output differs")
        previous = trans["state_after"]
    check(previous == oracle["final_state"], "final state differs from last observed transition")

    reports = {r["payload"]["actor"]: r["payload"]["content"] for r in journal
               if r["kind"] == "communication" and r["payload"]["target"] == "harness_report_store"}
    report_rows = [r for r in journal if r["kind"] == "communication" and r["payload"]["target"] == "harness_report_store"]
    check(len(reports) == len(report_rows), "duplicate final report from a stage")
    stages = indexed_files(folder / "stages", ".json")
    order = topological_order(case["workflow"])
    node_claim_paths = indexed_files(folder / "api/nodes", ".claim.json")
    check(len(stages) == len(node_claim_paths), "stage entry lacks corresponding executor claim")
    check(len(stages) <= len(order), "unexpected stages")
    config = collection.read(folder / "api/CONFIG.json")
    settings = plan["settings"]
    check(config["model_requested"] == settings["model"] == EXPECTED_MODEL, "model request configuration differs")
    for key in ("max_calls", "max_turns", "turn_timeout_seconds"):
        check(config[key] == settings[key], "executor budget differs: " + key)
    ms = config["model_settings"]
    check(ms["max_tokens"] == settings["max_output_tokens"] and ms["reasoning"]["effort"] == settings["reasoning"], "token/reasoning settings differ")
    check(ms["retry"]["max_retries"] == 0 and ms["parallel_tool_calls"] is False and ms["store"] is False, "SDK policy differs")
    calls = indexed_files(folder / "api/calls", ".claim.json")
    check(len(calls) <= settings["max_calls"], "episode call budget exceeded")
    claims = [collection.read(p) for p in calls]
    responses, errors, response_ids, last_claimed = {}, {}, [], None
    for number, (path, claim) in enumerate(zip(calls, claims), 1):
        check(claim["call"] == number and claim["schema"] == "sdk-model-call-claim", "invalid model claim")
        check(collection.digest(claim["request"]) == claim["request_sha256"], "request hash differs")
        when = datetime.fromisoformat(claim["claimed_at"])
        check(when.tzinfo is not None and (last_claimed is None or when >= last_claimed), "claim timestamps are not ordered")
        last_claimed = when
        for kind, destination in (("response", responses), ("error", errors)):
            receipt = path.with_name(f"{number:06d}.{kind}.json")
            if receipt.exists():
                value = collection.read(receipt)
                check((value["call"], value["node_id"], value["request_sha256"]) ==
                      (number, claim["node_id"], claim["request_sha256"]), "response/error does not bind to model request")
                check(value["seconds"] >= 0, "negative observed duration")
                destination[number] = value
        check(number in responses or number in errors, "claimed model call has no known response/error outcome")
        if number in responses:
            identity = responses[number]["response"].get("response_id")
            if identity:
                response_ids.append(identity)
            check((identity and response_ids.count(identity) == 1) or number in errors, "unflagged missing/duplicate response identity")
        if number in errors:
            check(errors[number]["response_receipt_exists"] == (number in responses), "error receipt availability differs")
    check(len(list((folder / "api/calls").glob("*.response.json"))) == len(responses), "orphan response receipt")
    check(len(list((folder / "api/calls").glob("*.error.json"))) == len(errors), "orphan model error receipt")

    by_provider_id = {}
    for row in starts:
        identity = row["payload"]["provider_call_id"]
        check(isinstance(identity, str) and identity not in by_provider_id, "missing/duplicate callback call identity")
        by_provider_id[identity] = row
    matched_callbacks, requested_tools, unfinished_requests, consumed = set(), 0, 0, 0
    stage_summaries = []
    for index, (stage_path, claim_path) in enumerate(zip(stages, node_claim_paths)):
        stage, node_claim = collection.read(stage_path), collection.read(claim_path)
        node = order[index]
        check(stage["stage_id"] == node == node_claim["node_id"], "stage order/identity differs")
        seq = stage["sequence"]
        prefix = adapter.monitor_prefix(seq)
        check(stage["prefix_before"] == prefix and stage["prefix_sha256"] == collection.digest(prefix), "stage prefix includes future/unmatched evidence")
        packet = expected_packet(case, node, reports, prefix["tools"])
        check(stage["packet"] == packet == node_claim["packet"], "actual node packet differs from declared public task/parents")
        check(node_claim["packet_sha256"] == collection.digest(packet), "node packet hash differs")
        check(node_claim["calls_before"] == consumed, "node call boundary differs")
        for parent in [u for u, v in case["workflow"]["edges"] if v == node]:
            deliveries = [r for r in journal[:seq] if r["kind"] == "communication"
                          and r["payload"]["actor"] == parent and r["payload"]["target"] == node]
            check(len(deliveries) == 1 and deliveries[0]["payload"]["content"] == reports[parent], "predecessor content was not actually delivered before node entry")
        node_calls = [c for c in claims if c["node_id"] == node]
        check([c["call"] for c in node_calls] == list(range(consumed + 1, consumed + 1 + len(node_calls))), "cross-stage context/call interleaving")
        history = [{"role": "user", "content": canonical(packet["input"])}]
        node_error_path = claim_path.with_name(claim_path.name.replace(".claim.json", ".error.json"))
        node_response_path = claim_path.with_name(claim_path.name.replace(".claim.json", ".response.json"))
        check(node_error_path.exists() != node_response_path.exists(), "node needs exactly one terminal receipt")
        for model_claim in node_calls:
            number, req = model_claim["call"], model_claim["request"]
            check(req["model_requested"] == MODEL and req["model_settings"] == ms, "per-call model settings differ")
            check(req["system_instructions"] == packet["instructions"], "model instructions differ from public packet")
            check(req["previous_response_id"] is None and req["conversation_id"] is None, "remote hidden context reused")
            # Executor canonicalizes the packet before constructing schemas;
            # required-parameter order therefore follows sorted dictionary keys.
            declared = [{"name": t["name"], "description": t["description"], "parameters": parameter_schema(t)}
                        for t in json.loads(canonical(packet))["tools"]]
            check(req["tools"] == declared, "actual tool catalogue/schema differs")
            check(normalized(req["input"]) == normalized(history), "actual model history differs from initial task, replies and real tool feedback")
            if number not in responses:
                continue
            response = responses[number]["response"]
            outputs = response["output"]
            history += copy.deepcopy(outputs)
            for output in outputs:
                if output.get("type") != "function_call":
                    continue
                requested_tools += 1
                call_id = output["call_id"]
                actual = by_provider_id.get(call_id)
                if actual is None:
                    check(node_error_path.exists(), "completed node lost a requested tool callback")
                    unfinished_requests += 1
                    continue
                data = actual["payload"]
                check(call_id not in matched_callbacks and data["actor"] == node and data["tool"] == output["name"], "callback is not bound to the producing response/node")
                check(data["arguments"] == json.loads(output["arguments"]), "model tool arguments differ from actual callback")
                check(datetime.fromisoformat(model_claim["claimed_at"]) <= datetime.fromisoformat(actual["observed_at"]), "tool precedes its model request")
                check(actual["sequence"] > seq, "tool occurred before its node prefix")
                matched_callbacks.add(call_id)
                end = terminal[data["invocation_id"]]
                if end["kind"] == "tool_handler_returned":
                    history.append({"type": "function_call_output", "call_id": call_id, "output": end["payload"]["result"]})
                else:
                    check(node_error_path.exists(), "failed tool hidden by a completed node")
        consumed += len(node_calls)
        terminal_node = collection.read(node_error_path if node_error_path.exists() else node_response_path)
        check(terminal_node["node_id"] == node and terminal_node["calls_after"] == consumed, "node terminal call boundary differs")
        if node_response_path.exists():
            check(node_calls and node_calls[-1]["call"] in responses, "completed node lacks model response")
            report = terminal_node["report"]
            check(report == reports.get(node) == last_text(responses[node_calls[-1]["call"]]["response"]), "node report is not its actual assistant output")
            check(not any(c["call"] in errors for c in node_calls), "completed node contains failed request")
        else:
            check(index == len(stages)-1 and node not in reports and run["status"] == "failed", "execution continued after failed node")
        stage_summaries.append({"stage_id": node, "author_role": case["role_bindings"][node]["role"],
                                "calls": len(node_calls), "status": terminal_node["status"]})
    check(consumed == len(calls) and matched_callbacks == set(by_provider_id), "orphan requests or callbacks")
    if run["status"] == "completed":
        check(len(stages) == len(order) and len(reports) == len(order), "completed run skipped workflow nodes")
        check(run["executor_result"]["node_reports"] == reports, "returned reports differ from actual stage outputs")
        check(run["executor_result"]["tool_invocations"] == len(starts), "tool denominator differs")
    expected_starts = [{k: collection.read(p)[k] for k in ("stage_id", "sequence", "prefix_sha256")} for p in stages]
    check(run["stage_start_prefixes"] == expected_starts, "run stage prefix index differs")
    users = [r["payload"] for r in journal if r["kind"] == "communication" and r["payload"]["target"] == "user"]
    check([(p["actor"], p["content"]) for p in users] == [(n, reports[n]) for n in order if n in case["output_nodes"] and n in reports], "internal report incorrectly published or real user delivery missing")
    return {"case_id": case["case_id"], "status": run["status"], "api_attempts": len(calls),
        "api_responses": len(responses), "api_errors": len(errors), "response_ids": response_ids,
        "model_tool_requests": requested_tools, "unexecuted_model_tool_requests": unfinished_requests,
        "actual_tool_invocations": len(starts), "bank_dispatched": sum(t["bank_dispatched"] for t in transitions),
        "user_deliveries": len(users), "stages": stage_summaries, "checks": "passed"}


def audit(out):
    out = Path(out).resolve()
    # Refuse active acquisitions; do not inspect partial live responses.
    aggregate = collection.read(out / "COLLECTION.json")
    check(aggregate["status"] in {"completed", "completed_with_failures"}, "collection not finished")
    plan = collection.load_plan(out)
    check({name: importlib.metadata.version(name) for name in plan["environment"]["packages"]}
          == plan["environment"]["packages"], "audit dependency versions differ from acquisition freeze")
    check(aggregate["plan_sha256"] == plan["sha256"], "collection plan identity differs")
    results = [audit_case(out, plan, case) for case in plan["cases"]]
    check([(r["case_id"], r["status"]) for r in aggregate["cases"]] == [(r["case_id"], r["status"]) for r in results], "collection denominator differs")
    ids = [identity for row in results for identity in row["response_ids"]]
    check(len(ids) == len(set(ids)), "model response identity reused across acquisition")
    count = sum(row["api_attempts"] for row in results)
    check(count <= plan["maximum_api_calls"], "cohort API budget exceeded")
    inventory = {p.relative_to(out).as_posix(): collection.sha(p) for p in sorted(out.rglob("*"))
                 if p.is_file() and p.name != "ARTIFACT_AUDIT.json"}
    return {"schema": "harnessaudit-acquisition-artifact-audit-v1", "plan_sha256": plan["sha256"],
        "auditor_source_sha256": {name: collection.sha(collection.ROOT / name) for name in (
            'code/commands/audit_submission_harnessaudit.py', 'code/tests/test_submission_harnessaudit_audit.py')},
        "status": "passed", "api_attempts": count, "unique_response_ids": len(set(ids)), "cases": results,
        "artifact_sha256": inventory,
        "boundaries": ["No API call or outcome-based selection; full original task/role and policy-visible adaptation retained.",
            "Exact request/history checks ignore only SDK-expanded null metadata and output-only created_by.",
            "Receipts are SDK request/response records, not independent HTTP/server evidence or returned-model authentication.",
            "Private state chain consistency is checked; this is not an independent execution replay or semantic validation of every Bank mutation.",
            "Original actionwise labels recomputed unchanged; ec-t3 raw ID-format behavior is not repaired.",
            "No task success, settled payment, causal safety or temporal/combination labels are inferred."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=collection.DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    report = audit(args.out)
    if not args.check_only:
        collection.write_new(args.out / "ARTIFACT_AUDIT.json", report)
    print(json.dumps({k: report[k] for k in ("status", "plan_sha256", "api_attempts", "unique_response_ids")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
