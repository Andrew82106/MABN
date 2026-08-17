"""Offline-only tests.  MockTransport has no socket or Ollama implementation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p1v2_qualification_q1 import common
from p1v2_qualification_q1.common import read_jsonl, safe_relative_path, sha256_text, verify_source_contracts
from p1v2_qualification_q1.errors import PreflightError, TransportError
from p1v2_qualification_q1.protocol import build_chat_payload, load_tasks
from p1v2_qualification_q1.replay import replay_run
from p1v2_qualification_q1.runner import execute_attempt
from p1v2_qualification_q1.transport import ChatResult, MockTransport, ModelFingerprint, OllamaLoopbackTransport, VersionEvidence
from p1v2_qualification_q1.validation import validate_run


def _fingerprint() -> ModelFingerprint:
    return ModelFingerprint("qwen3:8b", "sha256:qualification-test", 123456, {"family": "qwen3", "parameter_size": "8B"})


def _version() -> VersionEvidence:
    return VersionEvidence("mock-ollama-version", 19, 0)


def _valid_result(task: dict) -> ChatResult:
    content = json.dumps(
        {
            "role": task["role"],
            "episode_id": task["episode_id"],
            "decision": "allow_continue",
            "public_report": task["expected_public_report"],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return ChatResult(content, sha256_text(content), len(content), False, None, 0)


def _transport(results: list[ChatResult | Exception]) -> MockTransport:
    return MockTransport([_fingerprint(), _fingerprint()], results, [_version(), _version()])


def _redirect_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_root = tmp_path / "q1_test_data"
    monkeypatch.setattr(common, "DATA_ROOT", data_root)
    monkeypatch.setattr(common, "RUNS_ROOT", data_root / "runs")


def _all_valid_results() -> list[ChatResult]:
    return [_valid_result(task) for task in load_tasks("screen") + load_tasks("confirmation")]


def test_full_mock_qualification_has_128_plus_two_and_replays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    transport = _transport(_all_valid_results())
    result = execute_attempt(transport, allow_live=True)
    assert result["decision"] == "qualified"
    assert transport.chat_calls == 128
    assert transport.tag_calls == 2
    validation = validate_run(result["run_id"])
    assert validation["passed"], validation
    replay = replay_run(result["run_id"])
    assert replay["passed"], replay
    validation_after_replay = validate_run(result["run_id"])
    assert validation_after_replay["passed"], validation_after_replay
    paths = result["paths"]
    assert paths["delivery"].is_file()
    assert len(read_jsonl(paths["public_sink"])) == 80
    first_payload = transport.payloads[0]
    assert first_payload == build_chat_payload(load_tasks("screen")[0])
    assert first_payload["model"] == "qwen3:8b"
    assert first_payload["format"] == "json" and first_payload["think"] is False and first_payload["stream"] is False


def test_screen_bad_output_blocks_confirmation_but_is_not_qualified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    tasks = load_tasks("screen")
    bad = ChatResult("not json", "bad-json", 8, False, None, 0)
    transport = _transport([bad, *[_valid_result(task) for task in tasks[1:]]])
    result = execute_attempt(transport, allow_live=True)
    assert result["decision"] == "not_qualified"
    assert transport.chat_calls == 32
    paths = result["paths"]
    assert len(read_jsonl(paths["unstarted"])) == 96
    validation = validate_run(result["run_id"])
    assert validation["passed"], validation


def test_environment_fault_has_priority_over_output_fault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    tasks = load_tasks("screen")
    bad = ChatResult("not json", "bad-json", 8, False, None, 0)
    results: list[ChatResult | Exception] = [bad, TransportError("timeout")]
    results.extend(_valid_result(task) for task in tasks[2:])
    transport = _transport(results)
    result = execute_attempt(transport, allow_live=True)
    assert result["decision"] == "rework"
    assert validate_run(result["run_id"])["passed"]


def test_preflight_tag_failure_creates_zero_inference_rework(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    transport = MockTransport([TransportError("tag_missing")], [], [_version()])
    result = execute_attempt(transport, allow_live=True)
    assert result["decision"] == "rework"
    assert transport.chat_calls == 0
    assert transport.tag_calls == 1
    assert validate_run(result["run_id"])["passed"]


def test_thinking_content_is_suppressed_and_never_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    secret = "do-not-write-this-reasoning"
    result0 = ChatResult(f"<think>{secret}</think>{{}}", "thinking-response", len(secret) + 18, False, None, 0)
    tasks = load_tasks("screen")
    transport = _transport([result0, *[_valid_result(task) for task in tasks[1:]]])
    result = execute_attempt(transport, allow_live=True)
    transcripts = result["paths"]["transcripts"].read_text(encoding="utf-8")
    assert secret not in transcripts
    assert result["decision"] == "not_qualified"
    assert validate_run(result["run_id"])["passed"]


def test_missing_authorization_is_pre_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    transport = _transport([])
    with pytest.raises(PreflightError, match="live_authorization_required"):
        execute_attempt(transport, allow_live=False)
    assert transport.chat_calls == 0 and transport.tag_calls == 0 and transport.version_calls == 0
    assert not (tmp_path / "q1_test_data").exists()


def test_tampered_ledger_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    transport = _transport(_all_valid_results())
    result = execute_attempt(transport, allow_live=True)
    transcript_path = result["paths"]["transcripts"]
    text = transcript_path.read_text(encoding="utf-8")
    transcript_path.write_text(text.replace("VQ-001", "VQ-999", 1), encoding="utf-8", newline="\n")
    validation = validate_run(result["run_id"])
    assert not validation["passed"]
    assert any("artifact_hash_mismatch:transcripts" == error or "accepted_invalid_model_output" == error for error in validation["errors"])


def test_frozen_source_contracts_are_exact_and_no_q0_python_import() -> None:
    contracts = verify_source_contracts()
    assert len(contracts) == 7
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (common.CODE_ROOT / "src" / "p1v2_qualification_q1").glob("*.py")
    )
    assert "import p1v2_qualification" not in source_text
    assert "from p1v2_qualification import" not in source_text


def test_invalid_paths_and_nonfrozen_payload_are_rejected_pre_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("../outside", "C:/outside", "//server/share", ".env", "folder/.env"):
        with pytest.raises(Exception):
            safe_relative_path(tmp_path, value, "test")
    adapter = OllamaLoopbackTransport()
    called = {"value": False}

    def forbidden_request(*args, **kwargs):
        called["value"] = True
        raise AssertionError("network must not be reached")

    monkeypatch.setattr(adapter, "_request", forbidden_request)
    with pytest.raises(TransportError, match="chat_payload_not_frozen"):
        adapter.chat({"model": "other"}, timeout=1)
    assert not called["value"]


def test_fingerprint_drift_is_rework_even_with_128_valid_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    changed = ModelFingerprint("qwen3:8b", "sha256:changed", 123456, {"family": "qwen3", "parameter_size": "8B"})
    transport = MockTransport([_fingerprint(), changed], _all_valid_results(), [_version(), _version()])
    result = execute_attempt(transport, allow_live=True)
    assert result["decision"] == "rework"
    assert validate_run(result["run_id"])["passed"]


def test_delivery_report_tampering_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_data_root(monkeypatch, tmp_path)
    result = execute_attempt(_transport(_all_valid_results()), allow_live=True)
    assert replay_run(result["run_id"])["passed"]
    result["paths"]["delivery"].write_text("tampered\n", encoding="utf-8", newline="\n")
    validation = validate_run(result["run_id"])
    assert not validation["passed"]
    assert "delivery_report_tampered" in validation["errors"]


def test_stage_wall_clock_marks_unstarted_and_reworks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _redirect_data_root(monkeypatch, tmp_path)

    class StepClock:
        def __init__(self) -> None:
            self.values = [0.0, 0.0, 0.0, 1201.0]

        def __call__(self) -> float:
            return self.values.pop(0) if self.values else 1201.0

    result = execute_attempt(_transport(_all_valid_results()), allow_live=True, clock=StepClock())
    assert result["decision"] == "rework"
    outcome = json.loads(result["paths"]["outcome"].read_text(encoding="utf-8"))
    assert outcome["screen_executed"] == 1
    assert outcome["screen_unstarted"] == 31
    assert outcome["confirmation_unstarted"] == 96
    assert validate_run(result["run_id"])["passed"]
