import asyncio
import copy
import json
import inspect
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'code/src'),
               str(Path(__file__).resolve().parents[2] / 'code/commands')]
import collect_contract_forecast_judge as module
from collect_contract_forecast_judge import schedule
from paperalpha_monitor.contract_forecast_llm import KEYS


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    original = socket.socket.connect
    socketpair_code = getattr(getattr(socket, "_fallback_socketpair", None), "__code__", None)
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden in judge fake tests")
    def guarded(sock, address):
        if socketpair_code is not None and inspect.currentframe().f_back.f_code is socketpair_code:
            return original(sock, address)
        return forbidden(sock, address)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", guarded)


def episode():
    p = {"schema": "paired-contract-forecast-observer-v1", "covered": True,
         "metadata": {"stage_id": "trade", "episode_id": "opaque"},
         "current_a": 0, "features": {"bias": 1}, "evidence_path": {}, "visible_prefix": {}}
    return {"episode_id": "opaque", "task_id": "fin-t1", "windows": [
        {"claim_number": 1, "covered": True, "projection": p}, {"claim_number": 2, "covered": False}]}


def test_only_permitted_prefixes_and_all_covered_windows():
    source = episode()
    original = copy.deepcopy(source)
    rows = schedule([source], {"fin-t1": {"id": "bound_risk_update_before_trade"}})
    assert len(rows) == 1 and rows[0]["claim_number"] == 1
    assert "opaque" not in rows[0]["request"]["input"][1]["content"]
    assert original == source


def test_duplicate_episode_and_window_rejected():
    e = episode()
    contracts = {"fin-t1": {"id": "bound_risk_update_before_trade"}}
    with pytest.raises(ValueError, match="duplicate public episode"):
        schedule([e, e], contracts)
    e["windows"].append(copy.deepcopy(e["windows"][0]))
    with pytest.raises(ValueError, match="duplicate public request"):
        schedule([e], contracts)


def dataset(tmp_path, windows):
    """Only handwritten public prefixes; the private sentinel is never input."""
    folder = tmp_path / "dataset"
    source = episode()
    prototype = source["windows"][0]
    source["windows"] = [dict(copy.deepcopy(prototype), claim_number=i)
                         for i in range(1, windows + 1)]
    # An uncovered request stays in the dataset but consumes no judge call.
    source["windows"].append({"claim_number": windows + 1, "covered": False})
    public = folder / "public/opaque.json"
    module.write_new(public, source)
    module.write_new(folder / "PLAN.json", {"episode_ids": [source["episode_id"]],
        "contracts": {"fin-t1": {"id": "bound_risk_update_before_trade"}}})
    secret = folder / "private/targets.json"
    module.write_new(secret, {"future_label": "MUST NEVER SELECT OR PROMPT FROM THIS"})
    module.write_new(folder / "private/REPORT.json", {"status": "projected", "artifact_sha256": {
        "public/opaque.json": module.sha(public), "private/targets.json": module.sha(secret)}})
    return folder


def probability_text(*, hard_a=0, total=1.0):
    probabilities = {key: 0.0 for key in KEYS}
    probabilities[f"{hard_a}|STOP|0|0"] = total
    return json.dumps({"probabilities": probabilities})


def response(identity="response-ok", *, text=None, **changes):
    value = {"id": identity, "model": "gpt-5.5", "status": "completed", "error": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "output": [{"type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": probability_text() if text is None else text}]}]}
    value.update(changes)
    return value


class FakeAPI:
    """Runs the real collector loop, asserting receipts before fake dispatch."""
    def __init__(self, output, replies):
        self.output, self.replies = output, replies
        self.responses = self
        self.requests = []
        self.active = self.maximum_active = 0
        self.closed = False

    async def create(self, **request):
        index = len(self.requests) + 1
        self.requests.append(copy.deepcopy(request))
        claim = module.read(self.output / "RUN.claim.json")
        receipt = module.read(self.output / "runs" / f"{index:04d}" / "REQUEST.json")
        assert receipt["plan_sha256"] == claim["plan_sha256"]
        assert receipt["request"] == request
        assert receipt["request_sha256"] == module.digest(request)
        assert receipt["claim_number"] == index
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            await asyncio.sleep(0)  # Let the second permitted request enter.
            item = self.replies[index - 1]
            if isinstance(item, BaseException):
                raise item
            return SimpleNamespace(model_dump=lambda **_: copy.deepcopy(item))
        finally:
            self.active -= 1

    async def close(self):
        self.closed = True


def test_real_run_loop_retains_all_failures_without_retry_or_private_labels(tmp_path, monkeypatch):
    refusal = response("refusal", output=[{"type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "refusal", "refusal": "synthetic refusal"}]}])
    replies = [response(), response(None), response(""), response("  "), response({"bad": "id"}),
        response("duplicate"), response("duplicate"), response("api-error", error={"code": "synthetic"}),
        TimeoutError("synthetic timeout"), refusal, response("invalid-json", text="not JSON"),
        response("incomplete", status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
        response("bad-sum", text=probability_text(total=0.5)),
        response("hard-A-conflict", text=probability_text(hard_a=1))]
    inputs, output = dataset(tmp_path, len(replies)), tmp_path / "judge"
    original_read = module.read
    def public_only_read(path):
        assert Path(path) != inputs / "private/targets.json"
        return original_read(path)
    monkeypatch.setattr(module, "read", public_only_read)
    prepared = module.prepare(inputs, output)
    assert prepared == {"prepared": True, "scheduled_api_calls": len(replies), "api_calls": 0}
    fake = FakeAPI(output, replies)
    factories = []
    def factory(endpoint, key):
        assert endpoint == module.API["endpoint"] and key == "synthetic-not-a-secret"
        factories.append(1)
        return fake
    monkeypatch.setenv("PAPERALPHA_LOCAL_API_KEY", "synthetic-not-a-secret")
    monkeypatch.setattr(module, "create_api_client", factory)
    summary = asyncio.run(module.run(output))
    assert summary["api_attempts"] == len(replies) == len(fake.requests)
    assert fake.closed and fake.maximum_active == module.API["concurrency"] == 2
    assert summary["outcomes"] == {"predicted": 1, "failed": len(replies) - 1}
    report = module.read(output / "REPORT.json")
    assert report["records"][0]["prediction"]["system_risk"] == 0
    for record in report["records"][1:]:
        assert record["status"] == "failed" and "prediction" not in record
    assert report["records"][8]["error_type"] == "TimeoutError"
    identity = module.read(output / "IDENTITY_AUDIT.json")
    assert identity["duplicate_response_ids"] == ["duplicate"]
    assert identity["affected_indices"] == [6, 7]
    for index in (6, 7):
        assert report["records"][index - 1]["error_type"] == "DuplicateResponseIdentity"
        # The immutable per-call parse is provisional; final acceptance is the report.
        assert module.read(output / "runs" / f"{index:04d}" / "RESULT.json")["status"] == "predicted"
    for index, reply in enumerate(replies, 1):
        folder = output / "runs" / f"{index:04d}"
        assert (folder / "REQUEST.json").is_file() and (folder / "RESULT.json").is_file()
        assert (folder / "RESPONSE.json").exists() is not isinstance(reply, BaseException)
    assert module.read(output / "runs/0008/RESPONSE.json")["error"] == {"code": "synthetic"}
    assert report["performance_evaluated"] is False and report["labels_read"] is False
    assert all("MUST NEVER" not in json.dumps(request) for request in fake.requests)
    with pytest.raises(FileExistsError):
        asyncio.run(module.run(output))
    assert len(factories) == 1 and len(fake.requests) == len(replies)


@pytest.mark.parametrize("mutation", ["schema", "api", "source", "budget", "request", "selected_subset"])
def test_rehashed_plan_cannot_change_profile_or_select_windows(tmp_path, mutation):
    inputs, output = dataset(tmp_path, 2), tmp_path / "judge"
    module.prepare(inputs, output)
    plan = module.read(output / "PLAN.json")
    if mutation == "schema": plan["schema"] = "other-profile"
    elif mutation == "api": plan["api"]["transport_retries"] = 1
    elif mutation == "source": plan["source_sha256"].pop(next(iter(plan["source_sha256"])))
    elif mutation == "budget": plan["maximum_api_calls"] = 577
    elif mutation == "request": plan["cases"][0]["request"]["model"] = "changed-model"
    else:
        plan["cases"].pop(); plan["maximum_api_calls"] = len(plan["cases"])
    plan["sha256"] = module.digest({key: value for key, value in plan.items() if key != "sha256"})
    (output / "PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_plan(output)


def test_prepare_rejects_more_than_576_calls_before_creating_output(tmp_path):
    inputs, output = dataset(tmp_path, 577), tmp_path / "judge"
    with pytest.raises(ValueError, match="bounded"):
        module.prepare(inputs, output)
    assert not output.exists()
