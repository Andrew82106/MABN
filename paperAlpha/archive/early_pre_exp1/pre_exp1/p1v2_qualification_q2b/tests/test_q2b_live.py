from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1v2_qualification_q2b.common import DATA_ROOT, canonical_json, load_json, sha256_file, write_json
from p1v2_qualification_q2b.env_config import FROZEN_BASE_URL, FROZEN_MODEL, LiveConfig, parse_dotenv_text, validate_live_values
from p1v2_qualification_q2b.errors import ConfigError, ContractViolation, TransportError
from p1v2_qualification_q2b.protocol import adapt_completion, build_request, load_frozen_assets, validate_frozen_assets
from p1v2_qualification_q2b.replay import replay_run
from p1v2_qualification_q2b.runner import run_staged
from p1v2_qualification_q2b.transport import RestrictedGatewayTransport, WireResponse
from p1v2_qualification_q2b.validation import validate_run


LITERAL_RUN_ID = "P1V2Q2B-REMOTE-20260802T000000000000Z"


class FakeResponse:
    def __init__(self, status: int, body: bytes, headers: list[tuple[str, str]] | None = None) -> None:
        self.status = status
        self._body = body
        self._headers = headers or []

    def read(self) -> bytes:
        return self._body

    @property
    def body(self) -> bytes:
        return self._body

    def getheaders(self) -> list[tuple[str, str]]:
        return self._headers


class FakeConnection:
    def __init__(self, response: FakeResponse | BaseException, factory: "ScriptedFactory") -> None:
        self._response = response
        self._factory = factory
        self.closed = False

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> None:
        if isinstance(self._response, BaseException):
            raise self._response
        self._factory.requests.append({"method": method, "path": path, "body": body, "headers": dict(headers or {})})

    def getresponse(self) -> FakeResponse:
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response

    def close(self) -> None:
        self.closed = True


class ScriptedFactory:
    def __init__(self, responses: list[FakeResponse | BaseException]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []
        self.connections: list[FakeConnection] = []

    def __call__(self) -> FakeConnection:
        if not self.responses:
            raise AssertionError("offline_script_exhausted")
        connection = FakeConnection(self.responses.pop(0), self)
        self.connections.append(connection)
        return connection


def _models_response(extra: bool = False) -> FakeResponse:
    models = [{"id": FROZEN_MODEL}]
    if extra:
        models.append({"id": "other-model"})
    return FakeResponse(200, canonical_json({"data": models}).encode("utf-8"))


def _completion_response(task: dict[str, object], serial: int) -> FakeResponse:
    content = {
        "agent_role": task["agent_role"],
        "episode_id": task["episode_id"],
        "decision": task["expected_decision"],
        "task_value": task["task_value"],
    }
    envelope = {
        "choices": [{"message": {"content": canonical_json(content)}}],
        "model": FROZEN_MODEL,
        "system_fingerprint": "not_provided",
        "provider_version": "not_provided",
    }
    return FakeResponse(200, canonical_json(envelope).encode("utf-8"), [("x-request-id", "mock-" + str(serial))])


def _config(marker: str = "test-key") -> LiveConfig:
    return LiveConfig(base_url=FROZEN_BASE_URL, api_key=marker, model=FROZEN_MODEL)


def _success_transport(tasks: list[dict[str, object]], marker: str = "test-key", *, end_extra_model: bool = False) -> tuple[RestrictedGatewayTransport, ScriptedFactory]:
    responses: list[FakeResponse | BaseException] = [_models_response()]
    responses.extend(_completion_response(task, index + 1) for index, task in enumerate(tasks))
    responses.append(_models_response(extra=end_extra_model))
    factory = ScriptedFactory(responses)
    return RestrictedGatewayTransport(_config(marker), factory), factory


class Q2BOfflineTests(unittest.TestCase):
    def _temporary_root(self):
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=DATA_ROOT)

    def _run_success(self, root: Path) -> dict[str, object]:
        assets = load_frozen_assets(ROOT)
        transport, _ = _success_transport(assets.screen_tasks + assets.confirmation_tasks)
        return run_staged(transport, data_root=root / "data", run_id=LITERAL_RUN_ID)

    def _sync_hash(self, run_dir: Path, data_root: Path, relative: str) -> None:
        manifest_path = run_dir / "manifest.json"
        manifest = load_json(manifest_path)
        target = data_root / "reports" / "DELIVERY.md" if relative == "reports/DELIVERY.md" else run_dir / relative
        manifest["artifact_sha256"][relative] = sha256_file(target)
        write_json(manifest_path, manifest)

    def test_injected_env_parser_rejects_missing_and_drift(self) -> None:
        values = parse_dotenv_text("P1V2_Q2_BASE_URL=http://127.0.0.1:58661/v1\nP1V2_Q2_API_KEY=fake\nP1V2_Q2_MODEL=gpt-5.3-codex-spark\n")
        config = validate_live_values(values)
        self.assertEqual(config.base_url, FROZEN_BASE_URL)
        with self.assertRaises(ConfigError):
            parse_dotenv_text("P1V2_Q2_BASE_URL=x\n")
        wrong_base = dict(values)
        wrong_base["P1V2_Q2_BASE_URL"] = "http://127.0.0.1:9999/v1"
        with self.assertRaises(ConfigError):
            validate_live_values(wrong_base)
        wrong_model = dict(values)
        wrong_model["P1V2_Q2_MODEL"] = "different-model"
        with self.assertRaises(ConfigError):
            validate_live_values(wrong_model)

    def test_one_frozen_request_profile(self) -> None:
        assets = load_frozen_assets(ROOT)
        validate_frozen_assets(assets)
        request = build_request(assets.screen_tasks[0], assets)
        self.assertEqual(set(request), {"model", "stream", "messages", "response_format"})
        self.assertEqual(request["model"], FROZEN_MODEL)
        self.assertIs(request["stream"], False)
        self.assertEqual(len(request["messages"]), 1)
        self.assertEqual(request["messages"][0]["role"], "user")
        self.assertNotIn("temperature", request)
        self.assertNotIn("top_p", request)
        self.assertNotIn("max_tokens", request)
        self.assertNotIn("seed", request)

    def test_restricted_transport_allowlist_redirect_and_http_errors(self) -> None:
        for status, expected in ((301, "http_redirect"), (401, "http_401"), (403, "http_403"), (429, "http_429"), (500, "http_5xx")):
            with self.subTest(status=status):
                factory = ScriptedFactory([FakeResponse(status, b"ignored")])
                transport = RestrictedGatewayTransport(_config(), factory)
                with self.assertRaises(TransportError) as raised:
                    transport.get_models()
                self.assertEqual(raised.exception.code, expected)
                self.assertEqual(transport.counts["metadata_calls"], 1)
                self.assertTrue(factory.connections[0].closed)
        timeout_factory = ScriptedFactory([TimeoutError()])
        with self.assertRaises(TransportError) as raised:
            RestrictedGatewayTransport(_config(), timeout_factory).get_models()
        self.assertEqual(raised.exception.code, "timeout")
        with self.assertRaises(TransportError) as raised:
            RestrictedGatewayTransport(_config(), ScriptedFactory([]))._perform("GET", "/forbidden", None)
        self.assertEqual(raised.exception.code, "endpoint_not_allowed")

    def test_adapter_rejects_envelope_content_and_semantic_errors(self) -> None:
        assets = load_frozen_assets(ROOT)
        task = assets.screen_tasks[0]
        valid = _completion_response(task, 1)
        cases: list[tuple[bytes, str, str]] = []
        malformed = json.loads(valid.body.decode("utf-8"))
        malformed["choices"][0]["message"]["content"] = "{"
        cases.append((canonical_json(malformed).encode("utf-8"), "response_content_json", "model_failure"))
        for field, value, code in (("agent_role", "publisher", "response_agent_role"), ("episode_id", "P1V2Q2B-SCREEN-999", "response_episode_id"), ("decision", "publish_safe", "response_decision"), ("task_value", "wrong", "response_task_value")):
            changed = json.loads(valid.body.decode("utf-8"))
            content = json.loads(changed["choices"][0]["message"]["content"])
            content[field] = value
            changed["choices"][0]["message"]["content"] = canonical_json(content)
            cases.append((canonical_json(changed).encode("utf-8"), code, "model_failure"))
        cases.extend(
            [
                (canonical_json({"model": FROZEN_MODEL, "choices": [{"message": {"content": ""}}]}).encode("utf-8"), "empty_content", "model_failure"),
                (canonical_json({"model": FROZEN_MODEL, "choices": [{"message": {"tool_calls": []}}]}).encode("utf-8"), "tool_calls", "model_failure"),
                (canonical_json({"model": FROZEN_MODEL, "choices": [{"message": {"function_call": {}}}]}).encode("utf-8"), "function_calls", "model_failure"),
                (canonical_json({"model": FROZEN_MODEL, "choices": [{"message": {"refusal": "no"}}]}).encode("utf-8"), "refusal", "model_failure"),
                (canonical_json({"model": FROZEN_MODEL, "choices": [{"message": {"reasoning_content": "hidden", "content": "{}"}}]}).encode("utf-8"), "reasoning_detected", "model_failure"),
                (canonical_json({"model": "different-model", "choices": [{"message": {"content": "{}"}}]}).encode("utf-8"), "model_mismatch", "infrastructure_failure"),
            ]
        )
        for body, code, disposition in cases:
            with self.subTest(code=code):
                with self.assertRaises(ContractViolation) as raised:
                    adapt_completion(body, {}, task, assets)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.disposition, disposition)

    def test_success_stages_validate_and_replay_without_socket(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            result = self._run_success(root)
            self.assertEqual(result["decision"], "qualified")
            self.assertEqual(result["transport_counts"]["metadata_calls"], 2)
            self.assertEqual(result["transport_counts"]["completion_calls"], 128)
            self.assertTrue(validate_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])
            self.assertTrue(replay_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])

    def test_screen_contract_failure_completes_gate_and_blocks_downstream(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            assets = load_frozen_assets(ROOT)
            bad = _completion_response(assets.screen_tasks[0], 1)
            envelope = json.loads(bad.body.decode("utf-8"))
            envelope["choices"][0]["message"]["content"] = "{"
            responses: list[FakeResponse | BaseException] = [_models_response(), FakeResponse(200, canonical_json(envelope).encode("utf-8"))]
            responses.extend(_completion_response(task, index + 2) for index, task in enumerate(assets.screen_tasks[1:8]))
            responses.append(_models_response())
            transport = RestrictedGatewayTransport(_config(), ScriptedFactory(responses))
            result = run_staged(transport, data_root=root / "data", run_id=LITERAL_RUN_ID)
            transcript = load_json(Path(result["run_dir"]) / "transcript.json")
            self.assertEqual(result["decision"], "not_qualified")
            self.assertEqual(result["transport_counts"]["completion_calls"], 8)
            self.assertEqual(result["transport_counts"]["metadata_calls"], 2)
            self.assertTrue(all(record["status"] == "unstarted" for record in transcript["records"][8:]))
            self.assertTrue(validate_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])
            self.assertTrue(replay_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])

    def test_infrastructure_failure_immediately_stops(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            factory = ScriptedFactory([_models_response(), FakeResponse(429, b"server-body-must-not-persist")])
            transport = RestrictedGatewayTransport(_config(), factory)
            result = run_staged(transport, data_root=root / "data", run_id=LITERAL_RUN_ID)
            self.assertEqual(result["decision"], "rework")
            self.assertEqual(result["transport_counts"]["metadata_calls"], 1)
            self.assertEqual(result["transport_counts"]["completion_calls"], 1)
            self.assertTrue(validate_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])
            self.assertTrue(replay_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])
            text = "\n".join(path.read_text(encoding="utf-8") for path in (root / "data").rglob("*") if path.is_file())
            self.assertNotIn("server-body-must-not-persist", text)

    def test_end_identity_drift_is_rework(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            assets = load_frozen_assets(ROOT)
            transport, _ = _success_transport(assets.screen_tasks + assets.confirmation_tasks, end_extra_model=True)
            result = run_staged(transport, data_root=root / "data", run_id=LITERAL_RUN_ID)
            self.assertEqual(result["decision"], "rework")
            self.assertTrue(validate_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])
            self.assertTrue(replay_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])

    def test_fake_key_and_authorization_do_not_reach_artifacts(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            assets = load_frozen_assets(ROOT)
            marker = "".join(("isolated", "_fake", "_credential"))
            transport, factory = _success_transport(assets.screen_tasks + assets.confirmation_tasks, marker)
            result = run_staged(transport, data_root=root / "data", run_id=LITERAL_RUN_ID)
            artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in (root / "data").rglob("*") if path.is_file())
            self.assertNotIn(marker, artifact_text)
            self.assertNotIn("authorization", artifact_text.lower())
            self.assertTrue(any("Authorization" in request["headers"] for request in factory.requests))

    def test_tampering_and_offline_verifiers_are_detected_and_isolated(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            result = self._run_success(root)
            data_root = root / "data"
            run_dir = Path(result["run_dir"])
            outcome = load_json(run_dir / "outcome.json")
            outcome["transport_counts"]["retry_count"] = 1
            write_json(run_dir / "outcome.json", outcome)
            self._sync_hash(run_dir, data_root, "outcome.json")
            self.assertFalse(validate_run(LITERAL_RUN_ID, data_root=data_root)["passed"])
        validation_source = (ROOT / "src" / "p1v2_qualification_q2b" / "validation.py").read_text(encoding="utf-8")
        replay_source = (ROOT / "src" / "p1v2_qualification_q2b" / "replay.py").read_text(encoding="utf-8")
        initializer = (ROOT / "src" / "p1v2_qualification_q2b" / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn(".transport", validation_source)
        self.assertNotIn(".transport", replay_source)
        self.assertNotIn(".env", validation_source)
        self.assertNotIn(".env", replay_source)
        self.assertNotIn("from .runner", initializer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
