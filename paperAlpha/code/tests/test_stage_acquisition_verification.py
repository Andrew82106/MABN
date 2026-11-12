import asyncio
import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
sys.path.insert(0, str(ROOT / 'code/commands'))
from paperalpha_runtime.stage_workflow import execute_stages
from paperalpha_runtime.stage_capture import stage_instructions
from verify_stage_acquisition import check_execution
from test_stage_workflow import workflow


def fixture():
    w = workflow()
    actors = {a: {"role": a, "description": "duty", "system_prompt": "original instructions"} for a in ("hub", "worker")}
    selected = {"stage_workflow": w, "public_configuration": {"goal": "goal", "agents": list(actors.values())}}
    manifest = {"stage_workflow": w, "status": "completed", "model_requested": "test",
                "settings": {"reasoning": "low", "max_output_tokens": 1800}}
    journal, stages, requests, responses = [], [], [], []
    def emit(kind, payload):
        journal.append({"schema_version": "runtime-journal-v1", "sequence": len(journal) + 1, "kind": kind, "payload": payload})
    emit("run_started", {"agents": list(actors), "external_endpoints": ["user"], "declared_stage_workflow": w,
                         "public_configuration": selected["public_configuration"]})
    async def execute(stage, actor, content):
        call = len(requests) + 1
        prompt = stage_instructions(actors[actor], stage, w["purposes"][stage])
        stages.append({"stage_id": stage, "actor": actor, "status": "started", "first_possible_call": call,
                       "journal_before": len(journal), "input": content, "instructions": prompt})
        requests.append({"call": call, "stage_id": stage, "actor": actor, "model_requested": "test", "system_instructions": prompt,
            "input": [{"role": "user", "content": content}], "conversation_id": None, "previous_response_id": None,
            "model_settings": {"reasoning": {"effort": "low"}, "max_tokens": 1800, "parallel_tool_calls": False, "store": False}})
        binding = {"actor": actor, "stage_id": stage, "invocation_id": str(call), "provider_call_id": str(call), "tool": "read"}
        emit("tool_handler_started", {**binding, "arguments": {}})
        emit("tool_handler_returned", {**binding, "result": "result"})
        output = stage + " report"
        responses.append({"call": call, "response": {"output": [{"type": "message", "content": [{"type": "output_text", "text": output}]}]}})
        stages.append({"stage_id": stage, "actor": actor, "status": "completed", "calls_started": 1,
                       "last_call": call, "journal_after": len(journal), "output": output})
        return output
    outputs = asyncio.run(execute_stages(w, "goal", execute, lambda d: emit("communication", d)))
    emit("communication", {"actor": "hub", "target": "user", "stage_id": "s:final", "content": outputs["s:final"]})
    emit("run_finished", {"status": "completed"})
    return selected, manifest, journal, stages, outputs, requests, responses


def test_verifier_binds_actual_model_text_inputs_and_tool_stage_intervals():
    report = check_execution(*fixture())
    assert report["completed_stages"] == 3 and report["physical_agents"] == 2
    assert report["internal_deliveries"] == 2 and report["fresh_first_inputs_verified"]


@pytest.mark.parametrize("mutate", [
    lambda x: x[5][0].update(input="unrecorded context"),
    lambda x: x[5][0].update(previous_response_id="another stage history"),
    lambda x: x[5][0].update(conversation_id="shared conversation"),
    lambda x: x[5][2].update(stage_id="s:first"),
    lambda x: x[5][1].update(actor="hub"),
    lambda x: x[5][0]["model_settings"].update(max_tokens=4096),
    lambda x: x[6][0]["response"]["output"][0]["content"][0].update(text="different model report"),
    lambda x: x[2][1]["payload"].update(stage_id="s:final"),
    lambda x: x[3][1].update(journal_after=1),
    lambda x: x[3].__setitem__(slice(0, 4), [x[3][0], x[3][2], x[3][1], x[3][3]]),
    lambda x: x[4].update(extra="fabricated output"),
])
def test_tampered_stage_context_request_output_or_scope_is_rejected(mutate):
    data = list(fixture())
    mutate(data)
    with pytest.raises((ValueError, KeyError)):
        check_execution(*data)


def test_partial_failed_run_never_becomes_completed_or_fabricates_reports():
    selected, manifest, journal, stages, outputs, requests, responses = fixture()
    manifest["status"] = "failed"
    end = {**stages[1], "status": "failed"}
    end.pop("output")
    partial = copy.deepcopy(journal[:3])
    partial.append({"schema_version": "runtime-journal-v1", "sequence": 4, "kind": "run_finished", "payload": {"status": "failed"}})
    result = check_execution(selected, manifest, partial, [stages[0], end], {}, requests[:1], responses[:1])
    assert result["completed_stages"] == 0 and result["started_stages"] == 1
