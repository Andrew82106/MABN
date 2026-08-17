from __future__ import annotations

import subprocess
import urllib.request
from dataclasses import replace
from types import SimpleNamespace

import pytest

from p1_benign.cli import main
from p1_benign.core import (
    DEFAULT_CONTEXT,
    EventRecorder,
    LIVE_MODEL,
    LOCAL_MODEL_SHAKEDOWN,
    TEST_DOUBLE,
)
from p1_benign.configuration import load_static_inputs
from p1_benign.providers import (
    DeterministicTestProvider,
    ModelProvider,
    ModelRequest,
    OllamaLocalProvider,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderUnavailableError,
    _safe_json_from_model_content,
    contains_reasoning_field,
    ollama_model_is_installed,
    sanitize_observable_response,
)
from p1_benign.readiness import check_live_readiness
from p1_benign.runner import run_p1
from p1_benign.workflow import execute_episode


def _intake_request() -> ModelRequest:
    return ModelRequest(
        run_id="P1-BENIGN-DRY-PROVIDER",
        episode_id="P1-BENIGN-DRY-PROVIDER-E001",
        role_id="intake",
        role_prompt="test",
        phase="produce_message",
        call_index=1,
        visible_messages=[
            {
                "source_agent": "task_input",
                "content": {
                    "vendor_id": "V-X",
                    "vendor_name": "Fictional",
                    "declared_registration_status": "active",
                    "public_risk_score": 1,
                    "evidence_reference": "FICT-X",
                },
            }
        ],
        response_contract={"type": "object"},
    )


def test_deterministic_provider_is_network_free(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    result = DeterministicTestProvider().generate(_intake_request())
    assert result.response["kind"] == "message"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:11434",
        "http://example.com:11434",
        "http://user:password@127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
    ],
)
def test_ollama_rejects_non_loopback_or_non_origin_endpoint(
    endpoint: str,
) -> None:
    with pytest.raises(ValueError):
        OllamaLocalProvider(endpoint=endpoint)


def test_ollama_accepts_loopback_and_fixes_model() -> None:
    provider = OllamaLocalProvider(
        endpoint="http://localhost:11434",
        model_id="qwen3:8b",
    )
    assert provider.model_id == "qwen3:8b"
    assert provider.concurrency == 1
    assert provider.decoding["thinking"] is False
    with pytest.raises(ValueError):
        OllamaLocalProvider(model_id="another-model")


def test_ollama_presence_check_only_runs_list(monkeypatch) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            stdout="NAME ID SIZE MODIFIED\nqwen3:8b abc 5GB now\n"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert ollama_model_is_installed("qwen3:8b") is True
    assert calls[0][0] == ["ollama", "list"]
    assert all(
        word not in calls[0][0]
        for word in ("pull", "create", "run")
    )


def test_ollama_presence_rejects_other_model_without_command(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("ollama command ran"),
    )
    assert ollama_model_is_installed("qwen3:latest") is False


def test_reasoning_tags_are_discarded_before_json_logging() -> None:
    parsed, discarded = _safe_json_from_model_content(
        '<think>private scratch</think>{"kind":"message","content":{}}'
    )
    assert parsed == {"kind": "message", "content": {}}
    assert discarded is True
    assert "private scratch" not in str(parsed)


def test_nested_reasoning_fields_are_recursively_discarded() -> None:
    cleaned, discarded = sanitize_observable_response(
        {
            "kind": "message",
            "reasoning": "private top level",
            "reasoningContent": "private camel case",
            "reasoning-details": "private hyphenated",
            "reasoningMetadata": "private unknown reasoning suffix",
            "scratch_pad": "private scratchpad",
            "content": {
                "vendor_id": "V-X",
                "nested": {
                    "reasoning_content": "private nested",
                    "rationale": "private rationale",
                    "rationaleText": "private rationale suffix",
                    "internalMonologue": "private internal monologue",
                    "deeper": [
                        {
                            "Analysis": "private case-insensitive analysis",
                            "safe_value": 7,
                        }
                    ],
                    "safe": True,
                },
            },
        }
    )
    assert discarded is True
    assert contains_reasoning_field(cleaned) is False
    assert "private" not in str(cleaned)
    assert cleaned["content"]["nested"] == {
        "deeper": [{"safe_value": 7}],
        "safe": True,
    }


def test_workflow_defense_discards_reasoning_before_event_log() -> None:
    class ReasoningProvider(DeterministicTestProvider):
        def generate(self, request):
            result = super().generate(request)
            response = dict(result.response)
            response["reasoning"] = "DO-NOT-LOG-TOP"
            response["reasoningContent"] = "DO-NOT-LOG-CAMEL"
            if response["kind"] == "message":
                response["content"] = dict(response["content"])
                response["content"]["reasoning_content"] = (
                    "DO-NOT-LOG-NESTED"
                )
                response["content"]["scratchpad"] = (
                    "DO-NOT-LOG-SCRATCHPAD"
                )
            return replace(
                result,
                response=response,
                safe_response_hash_source="DO-NOT-HASH-REASONING",
            )

    static = load_static_inputs(DEFAULT_CONTEXT)
    task_id = static.task_ids[0]
    recorder = EventRecorder(
        run_id="P1-BENIGN-DRY-REASONING",
        execution_mode=TEST_DOUBLE,
        eligible_for_scientific_analysis=False,
    )
    execute_episode(
        run_id="P1-BENIGN-DRY-REASONING",
        episode_number=1,
        task_instance_id=task_id,
        material=static.materials[task_id],
        internal_record=static.internal_records[task_id],
        expected_report=static.expected_reports[task_id]["report"],
        static=static,
        provider=ReasoningProvider(),
        recorder=recorder,
    )
    serialized = str(recorder.events)
    assert "DO-NOT-LOG" not in serialized
    assert "DO-NOT-HASH" not in serialized
    outputs = [
        event
        for event in recorder.events
        if event["event_type"] == "model_output"
    ]
    assert outputs
    assert all(
        event["payload"]["reasoning_discarded"] is True
        and not contains_reasoning_field(
            event["payload"]["parsed_response"]
        )
        for event in outputs
    )


def test_workflow_enforces_full_response_contract_before_logging() -> None:
    class ExtraFieldProvider(DeterministicTestProvider):
        def generate(self, request):
            result = super().generate(request)
            response = dict(result.response)
            response["unexpected_top_level"] = "DO-NOT-LOG-EXTRA"
            return replace(result, response=response)

    static = load_static_inputs(DEFAULT_CONTEXT)
    task_id = static.task_ids[0]
    recorder = EventRecorder(
        run_id="P1-BENIGN-DRY-CONTRACT",
        execution_mode=TEST_DOUBLE,
        eligible_for_scientific_analysis=False,
    )
    outcome = execute_episode(
        run_id="P1-BENIGN-DRY-CONTRACT",
        episode_number=1,
        task_instance_id=task_id,
        material=static.materials[task_id],
        internal_record=static.internal_records[task_id],
        expected_report=static.expected_reports[task_id]["report"],
        static=static,
        provider=ExtraFieldProvider(),
        recorder=recorder,
    )
    assert outcome["parse_failure"] is True
    assert not any(
        event["event_type"] == "model_output"
        for event in recorder.events
    )
    assert any(
        event["event_type"] == "model_call_failed"
        and event["payload"]["error_type"] == "ProviderParseError"
        for event in recorder.events
    )
    assert "DO-NOT-LOG-EXTRA" not in str(recorder.events)


@pytest.mark.parametrize(
    "result_changes",
    [
        {"latency_ms": float("nan")},
        {
            "token_usage": {
                "prompt_tokens": -1,
                "completion_tokens": None,
                "total_tokens": None,
            }
        },
        {
            "token_usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 4,
            }
        },
        {"finish_reason": True},
        {"reasoning_discarded": 1},
    ],
)
def test_invalid_provider_metadata_becomes_one_parse_terminal(
    result_changes,
) -> None:
    class InvalidMetadataProvider(DeterministicTestProvider):
        def generate(self, request):
            return replace(
                super().generate(request),
                **result_changes,
            )

    static = load_static_inputs(DEFAULT_CONTEXT)
    task_id = static.task_ids[0]
    recorder = EventRecorder(
        run_id="P1-BENIGN-DRY-BAD-METADATA",
        execution_mode=TEST_DOUBLE,
        eligible_for_scientific_analysis=False,
    )
    outcome = execute_episode(
        run_id="P1-BENIGN-DRY-BAD-METADATA",
        episode_number=1,
        task_instance_id=task_id,
        material=static.materials[task_id],
        internal_record=static.internal_records[task_id],
        expected_report=static.expected_reports[task_id]["report"],
        static=static,
        provider=InvalidMetadataProvider(),
        recorder=recorder,
    )
    requests = [
        event
        for event in recorder.events
        if event["event_type"] == "model_call_requested"
    ]
    terminals = [
        event
        for event in recorder.events
        if event["event_type"]
        in {"model_output", "model_call_failed"}
    ]
    assert outcome["parse_failure"] is True
    assert len(requests) == len(terminals) == 1
    assert terminals[0]["event_type"] == "model_call_failed"
    assert terminals[0]["payload"]["error_type"] == "ProviderParseError"


@pytest.mark.parametrize(
    ("exception", "expected_error"),
    [
        (ProviderError("declared failure"), "ProviderUnavailableError"),
        (
            ProviderUnavailableError("unavailable"),
            "ProviderUnavailableError",
        ),
        (RuntimeError("unexpected"), "UnexpectedProviderError"),
    ],
)
def test_provider_exceptions_normalize_to_one_valid_terminal(
    exception: Exception,
    expected_error: str,
) -> None:
    class FailingProvider(DeterministicTestProvider):
        def generate(self, request):
            raise exception

    static = load_static_inputs(DEFAULT_CONTEXT)
    task_id = static.task_ids[0]
    recorder = EventRecorder(
        run_id="P1-BENIGN-DRY-PROVIDER-FAIL",
        execution_mode=TEST_DOUBLE,
        eligible_for_scientific_analysis=False,
    )
    execute_episode(
        run_id="P1-BENIGN-DRY-PROVIDER-FAIL",
        episode_number=1,
        task_instance_id=task_id,
        material=static.materials[task_id],
        internal_record=static.internal_records[task_id],
        expected_report=static.expected_reports[task_id]["report"],
        static=static,
        provider=FailingProvider(),
        recorder=recorder,
    )
    requests = [
        event
        for event in recorder.events
        if event["event_type"] == "model_call_requested"
    ]
    terminals = [
        event
        for event in recorder.events
        if event["event_type"]
        in {"model_output", "model_call_failed"}
    ]
    assert len(requests) == len(terminals) == 1
    assert terminals[0]["payload"]["error_type"] == expected_error


def test_wrong_known_tool_reaches_permission_denial_measurement() -> None:
    class WrongToolProvider(DeterministicTestProvider):
        def generate(self, request):
            result = super().generate(request)
            if (
                request.role_id == "internal_record_agent"
                and request.phase == "request_tool"
            ):
                return replace(
                    result,
                    response={
                        "kind": "tool_call",
                        "tool_name": "external_sink.publish",
                        "arguments": {
                            "vendor_id": "V-P1-001",
                            "registration_status": "active",
                            "risk_level": "low",
                            "recommendation": "approve",
                        },
                    },
                )
            return result

    static = load_static_inputs(DEFAULT_CONTEXT)
    task_id = static.task_ids[0]
    recorder = EventRecorder(
        run_id="P1-BENIGN-DRY-WRONG-TOOL",
        execution_mode=TEST_DOUBLE,
        eligible_for_scientific_analysis=False,
    )
    outcome = execute_episode(
        run_id="P1-BENIGN-DRY-WRONG-TOOL",
        episode_number=1,
        task_instance_id=task_id,
        material=static.materials[task_id],
        internal_record=static.internal_records[task_id],
        expected_report=static.expected_reports[task_id]["report"],
        static=static,
        provider=WrongToolProvider(),
        recorder=recorder,
    )
    denied = [
        event
        for event in recorder.events
        if event["event_type"] == "tool_call"
        and event["payload"]["permission_allowed"] is False
    ]
    assert outcome["unauthorized_tool_attempt"] is True
    assert outcome["parse_failure"] is False
    assert len(denied) == 1
    assert denied[0]["payload"]["result"] == {"status": "denied"}


def test_remote_provider_construction_does_not_read_credential(
    monkeypatch,
) -> None:
    monkeypatch.delenv("P1_FAKE_REMOTE_KEY", raising=False)
    provider = OpenAICompatibleProvider(
        provider_id="openai_compatible_remote",
        model_id="future-model",
        endpoint="https://example.invalid/v1",
        credential_env="P1_FAKE_REMOTE_KEY",
        decoding={
            "temperature": 0,
            "max_output_tokens": 256,
            "frozen": True,
        },
    )
    assert provider.manifest_summary()["credential_env"] == (
        "P1_FAKE_REMOTE_KEY"
    )


def test_live_readiness_fails_without_provider(test_context) -> None:
    result = check_live_readiness(
        context=test_context,
        allow_live=True,
        provider_id=None,
        model_id="future-model",
    )
    assert result["ready"] is False
    assert result["checks"]["provider_present"] is False
    assert result["network_request_attempted"] is False


def test_live_readiness_fails_without_model(test_context) -> None:
    result = check_live_readiness(
        context=test_context,
        allow_live=True,
        provider_id="openai_compatible_remote",
        model_id=None,
        endpoint="https://example.invalid/v1",
        environment={"P1_REMOTE_API_KEY": "FAKE-TEST-ONLY"},
    )
    assert result["ready"] is False
    assert result["checks"]["model_present"] is False


def test_live_readiness_fails_without_explicit_allow(test_context) -> None:
    result = check_live_readiness(
        context=test_context,
        allow_live=False,
        provider_id="openai_compatible_remote",
        model_id="future-model",
        endpoint="https://example.invalid/v1",
        environment={"P1_REMOTE_API_KEY": "FAKE-TEST-ONLY"},
    )
    assert result["ready"] is False
    assert result["checks"]["explicit_allow_live"] is False


def test_live_readiness_remains_blocked_until_all_freezes(
    test_context,
) -> None:
    result = check_live_readiness(
        context=test_context,
        allow_live=True,
        provider_id="openai_compatible_remote",
        model_id="future-model",
        endpoint="https://example.invalid/v1",
        environment={"P1_REMOTE_API_KEY": "FAKE-TEST-ONLY"},
    )
    assert result["ready"] is False
    assert result["checks"]["provider_frozen"] is False
    assert result["checks"]["model_frozen"] is False
    assert result["checks"]["decoding_frozen"] is False
    assert result["checks"]["budget_frozen"] is False
    assert result["checks"]["live_execution_enabled"] is False


def test_xfyun_rejected_before_credential_inspection(test_context) -> None:
    class ExplodingEnvironment(dict):
        def __contains__(self, key):
            raise AssertionError("credential environment was inspected")

    result = check_live_readiness(
        context=test_context,
        allow_live=True,
        provider_id="xfyun_astron_coding_plan",
        model_id="xsparkx2agent",
        endpoint=(
            "https://maas-coding-api.cn-huabei-1.xf-yun.com/v2"
        ),
        environment=ExplodingEnvironment(
            {"XFYUN_ASTRON_CODING_API_KEY": "FAKE"}
        ),
    )
    assert result["ready"] is False
    assert result["allowed_for_automated_experiment"] is False
    assert result["credential_checked"] is False
    assert result["network_request_attempted"] is False
    assert any("interactive_coding_only" in item for item in result["reasons"])


@pytest.mark.parametrize(
    "mode",
    [TEST_DOUBLE, LOCAL_MODEL_SHAKEDOWN],
)
def test_non_scientific_event_recorder_rejects_eligibility(
    mode: str,
) -> None:
    with pytest.raises(ValueError):
        EventRecorder(
            run_id=(
                "P1-BENIGN-DRY-ELIG"
                if mode == TEST_DOUBLE
                else "P1-BENIGN-LOCAL-ELIG"
            ),
            execution_mode=mode,
            eligible_for_scientific_analysis=True,
        )


def test_test_provider_cannot_be_scientifically_eligible(
    test_context,
) -> None:
    with pytest.raises(ValueError):
        run_p1(
            context=test_context,
            provider=DeterministicTestProvider(),
            execution_mode=TEST_DOUBLE,
            episode_count=1,
            run_id="P1-BENIGN-DRY-ELIGIBLE",
            eligible_for_scientific_analysis=True,
        )


def test_local_shakedown_cannot_exceed_one_episode(
    test_context,
) -> None:
    with pytest.raises(ValueError):
        run_p1(
            context=test_context,
            provider=OllamaLocalProvider(),
            execution_mode=LOCAL_MODEL_SHAKEDOWN,
            episode_count=2,
            run_id="P1-BENIGN-LOCAL-TOO-MANY",
            eligible_for_scientific_analysis=False,
        )


def test_runner_rejects_local_without_explicit_authorization(
    test_context,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "p1_benign.runner.ollama_model_is_installed",
        lambda model_id: pytest.fail("inventory check should not run"),
    )
    with pytest.raises(PermissionError):
        run_p1(
            context=test_context,
            provider=OllamaLocalProvider(),
            execution_mode=LOCAL_MODEL_SHAKEDOWN,
            episode_count=1,
            run_id="P1-BENIGN-LOCAL-NO-AUTH",
            eligible_for_scientific_analysis=False,
        )


def test_runner_rejects_nonfrozen_root_seed(test_context) -> None:
    with pytest.raises(ValueError):
        run_p1(
            context=test_context,
            provider=DeterministicTestProvider(),
            execution_mode=TEST_DOUBLE,
            episode_count=1,
            run_id="P1-BENIGN-DRY-WRONG-SEED",
            eligible_for_scientific_analysis=False,
            root_seed=1,
        )


def test_runner_rejects_nonfrozen_local_provider_before_request(
    test_context,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("local request attempted"),
    )
    with pytest.raises(ValueError):
        run_p1(
            context=test_context,
            provider=OllamaLocalProvider(
                timeout_seconds=5,
                max_output_tokens=512,
            ),
            execution_mode=LOCAL_MODEL_SHAKEDOWN,
            episode_count=1,
            run_id="P1-BENIGN-LOCAL-NONFROZEN",
            eligible_for_scientific_analysis=False,
            allow_local_shakedown=True,
        )


def test_runner_rejects_non_catalog_local_origin_before_inventory(
    test_context,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "p1_benign.runner.ollama_model_is_installed",
        lambda *args, **kwargs: pytest.fail(
            "model inventory reached before frozen-origin rejection"
        ),
    )
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("local request attempted"),
    )
    with pytest.raises(ValueError):
        run_p1(
            context=test_context,
            provider=OllamaLocalProvider(
                endpoint="http://127.0.0.1:9999",
            ),
            execution_mode=LOCAL_MODEL_SHAKEDOWN,
            episode_count=1,
            run_id="P1-BENIGN-LOCAL-WRONG-ORIGIN",
            eligible_for_scientific_analysis=False,
            allow_local_shakedown=True,
        )


def test_runner_rejects_live_api_even_with_remote_provider(
    test_context,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("remote request attempted"),
    )
    provider = OpenAICompatibleProvider(
        provider_id="openai_compatible_remote",
        model_id="future-model",
        endpoint="https://example.invalid/v1",
        credential_env="P1_FAKE_REMOTE_KEY",
        decoding={
            "temperature": 0,
            "sampling_seed": 20260731,
            "max_output_tokens": 256,
            "thinking": False,
            "frozen": True,
        },
    )
    with pytest.raises(PermissionError):
        run_p1(
            context=test_context,
            provider=provider,
            execution_mode=LIVE_MODEL,
            episode_count=1,
            run_id="P1-BENIGN-LIVE-DIRECT-BYPASS",
            eligible_for_scientific_analysis=True,
        )


def test_runner_rejects_test_provider_identity_impersonation(
    test_context,
) -> None:
    class ImposterProvider(ModelProvider):
        provider_id = "deterministic_test"
        model_id = "deterministic-p1-v1"
        endpoint_identifier = "none"
        decoding = DeterministicTestProvider.decoding
        concurrency = 1
        is_live_provider = True

        def generate(self, request):
            pytest.fail("imposter provider was called")

    with pytest.raises(ValueError):
        run_p1(
            context=test_context,
            provider=ImposterProvider(),
            execution_mode=TEST_DOUBLE,
            episode_count=1,
            run_id="P1-BENIGN-DRY-IMPOSTER",
            eligible_for_scientific_analysis=False,
        )


def test_run_live_cli_fails_safely_without_allow(
    test_context,
    capsys,
) -> None:
    code = main(
        [
            "run-p1-live",
            "--provider-id",
            "openai_compatible_remote",
            "--model-id",
            "future-model",
            "--endpoint",
            "https://example.invalid/v1",
        ],
        context=test_context,
    )
    output = capsys.readouterr().out
    assert code != 0
    assert '"network_request_attempted": false' in output


def test_local_cli_requires_explicit_allow(test_context, capsys) -> None:
    code = main(
        ["run-p1-local-shakedown"],
        context=test_context,
    )
    output = capsys.readouterr().out
    assert code != 0
    assert "explicit_local_shakedown_allow_required" in output
