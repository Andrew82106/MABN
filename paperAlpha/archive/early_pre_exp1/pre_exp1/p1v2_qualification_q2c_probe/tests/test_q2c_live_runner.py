from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from q2c_probe.config import (
    BASE_URL_IDENTITY,
    CARDS,
    LIVE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    TARGET_MODEL,
    canonical_json,
    profile_commitment,
    public_profile_snapshot,
    sha256_json,
)
from q2c_probe.evidence import (
    REQUIRED_ARTIFACTS,
    artifact_manifest,
    build_ledger,
    canonical_file_hash,
    derive_outcome,
    derive_report,
    local_delivery_text,
    transcript_semantic_view,
)
from q2c_probe.fake_transport import ManualClock
from q2c_probe.live_runner import (
    LIVE_GUARD_NAME,
    execute_live_probe_with_configuration,
    live_execution_source_hashes,
    reserve_live_attempt,
)
from q2c_probe.live_transport import LiveConfiguration, LiveHTTPTransport
from q2c_probe.replay import replay_run
from q2c_probe.runner import execute_protocol
from q2c_probe.validator import validate_run


LIVE_SECRET = "TEST_LIVE_SECRET_SENTINEL"
RAW_SECRET = "RAW_RESPONSE_SECRET_SENTINEL"


def models_payload(target_present: bool = True) -> dict[str, object]:
    model_id = TARGET_MODEL if target_present else "different-model"
    return {"data": [{"id": model_id}, {"id": "fake-secondary-model"}]}


def completion_payload(index: int, *, model: str = TARGET_MODEL) -> dict[str, object]:
    card = CARDS[index]
    content = card.expected_answer()
    if card.profile != "plain_text":
        content = canonical_json(
            {"probe_id": card.expected_probe_id(), "answer": card.expected_answer()}
        )
    return {
        "id": f"live-fake-request-{index + 1}",
        "model": model,
        "system_fingerprint": "live-fake-fingerprint",
        "provider_version": "live-fake-provider",
        "choices": [{"message": {"content": content}}],
    }


@dataclass
class HTTPScenario:
    status: int = 200
    payload: object = None
    error: BaseException | None = None
    duration_seconds: float = 0.0


class FakeHTTPResponse:
    def __init__(self, scenario: HTTPScenario, clock: ManualClock) -> None:
        self.status = scenario.status
        self._payload = scenario.payload
        self._clock = clock
        self._duration_seconds = scenario.duration_seconds
        self._error = scenario.error

    def read(self, amt: int | None = None) -> bytes:
        self._clock.advance(self._duration_seconds)
        if self._error is not None:
            raise self._error
        if isinstance(self._payload, bytes):
            return self._payload
        return canonical_json(self._payload).encode("utf-8")


class FakeHTTPConnection:
    def __init__(self, scenario: HTTPScenario, factory: "FakeHTTPFactory") -> None:
        self._scenario = scenario
        self._factory = factory
        self.closed = False

    def request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._factory.requests.append(
            {
                "method": method,
                "url": url,
                "body": body,
                "headers": dict(headers or {}),
            }
        )
        if self._scenario.error is not None and not isinstance(self._scenario.error, TimeoutError):
            raise self._scenario.error

    def getresponse(self) -> FakeHTTPResponse:
        return FakeHTTPResponse(self._scenario, self._factory.clock)

    def close(self) -> None:
        self.closed = True


class FakeHTTPFactory:
    def __init__(self, scenarios: list[HTTPScenario], clock: ManualClock | None = None) -> None:
        self.scenarios = list(scenarios)
        self.clock = clock or ManualClock()
        self.connection_calls: list[tuple[str, int, float]] = []
        self.requests: list[dict[str, object]] = []

    def __call__(self, host: str, port: int, timeout_seconds: float) -> FakeHTTPConnection:
        self.connection_calls.append((host, port, timeout_seconds))
        if not self.scenarios:
            raise AssertionError("unexpected_fake_connection")
        return FakeHTTPConnection(self.scenarios.pop(0), self)


class Q2CLiveRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp.name) / "data"
        self.config = LiveConfiguration(BASE_URL_IDENTITY, LIVE_SECRET, TARGET_MODEL)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def success_scenarios(self) -> list[HTTPScenario]:
        return [HTTPScenario(payload=models_payload())] + [
            HTTPScenario(payload=completion_payload(index)) for index in range(4)
        ]

    def test_live_fake_connection_sends_exact_frozen_requests_and_validates(self) -> None:
        clock = ManualClock()
        factory = FakeHTTPFactory(self.success_scenarios(), clock)
        original_open = Path.open

        def no_real_env(path: Path, *args: object, **kwargs: object):
            if path.name == ".env":
                raise AssertionError("real_env_forbidden")
            return original_open(path, *args, **kwargs)

        with patch("socket.create_connection", side_effect=AssertionError("socket_forbidden")), patch.object(Path, "open", no_real_env):
            run_id, run_dir = execute_live_probe_with_configuration(
                self.data_root,
                self.config,
                connection_factory=factory,
                run_id="P1V2Q2C-PROBE-FAKE-SUCCESS",
                clock=clock,
            )
        self.assertEqual(len(factory.connection_calls), 5)
        self.assertEqual(factory.connection_calls, [("127.0.0.1", 58661, 30.0)] * 5)
        self.assertEqual(factory.requests[0]["method"], "GET")
        self.assertEqual(factory.requests[0]["url"], "/v1/models")
        self.assertIsNone(factory.requests[0]["body"])
        for index, card in enumerate(CARDS, start=1):
            request = factory.requests[index]
            self.assertEqual(request["method"], "POST")
            self.assertEqual(request["url"], "/v1/chat/completions")
            self.assertEqual(json.loads(request["body"].decode("utf-8")), card.request_payload())
            self.assertEqual(request["headers"]["Authorization"], "Bearer " + LIVE_SECRET)
            payload = json.loads(request["body"].decode("utf-8"))
            self.assertNotIn("tools", payload)
            self.assertNotIn("tool_choice", payload)
            self.assertNotIn("temperature", payload)
            self.assertNotIn("top_p", payload)
            self.assertNotIn("max_tokens", payload)
            self.assertNotIn("seed", payload)
        self.assertEqual(validate_run(self.data_root, run_id), [])
        self.assertEqual(replay_run(self.data_root, run_id), [])
        self.assertEqual(validate_run(self.data_root, run_id), [])
        outcome = json.loads((run_dir / "outcome.json").read_text(encoding="utf-8"))
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
        delivery = (run_dir / "reports" / "DELIVERY.md").read_text(encoding="utf-8")
        self.assertEqual(outcome["run_kind"], "live_probe")
        self.assertEqual(transcript["schema_version"], LIVE_SCHEMA_VERSION)
        self.assertEqual(outcome["schema_version"], LIVE_SCHEMA_VERSION)
        self.assertEqual(report["schema_version"], LIVE_SCHEMA_VERSION)
        self.assertEqual(report["report_kind"], "live_probe_safe_summary")
        self.assertIs(report["offline_only"], False)
        self.assertEqual(report["scope"], "one_constrained_q2c_live_probe")
        self.assertIn("one constrained Q2-C live probe", delivery)
        self.assertNotIn("offline fake", delivery.lower())
        self.assertNotIn("no network", delivery.lower())
        self.assertEqual(delivery, local_delivery_text(run_id, outcome))
        for artifact in run_dir.rglob("*"):
            if artifact.is_file():
                text = artifact.read_text(encoding="utf-8")
                self.assertNotIn(LIVE_SECRET, text)
                self.assertNotIn(RAW_SECRET, text)
                self.assertNotIn("Authorization", text)
        self.assertTrue((self.data_root / LIVE_GUARD_NAME).is_file())

    def test_nonconforming_model_content_continues_all_four_cards(self) -> None:
        tool = completion_payload(0)
        tool["choices"] = [{"message": {"tool_calls": [{"arguments": RAW_SECRET}]}}]
        refusal = completion_payload(1)
        refusal["choices"] = [{"message": {"refusal": "not available"}}]
        non_json = completion_payload(2)
        non_json["choices"] = [{"message": {"content": "not-json"}}]
        fields = completion_payload(3)
        fields["choices"] = [{"message": {"content": "{}"}}]
        factory = FakeHTTPFactory(
            [HTTPScenario(payload=models_payload())]
            + [HTTPScenario(payload=value) for value in (tool, refusal, non_json, fields)]
        )
        run_id, _ = execute_live_probe_with_configuration(
            self.data_root,
            self.config,
            connection_factory=factory,
            run_id="P1V2Q2C-PROBE-FAKE-OBSERVATIONS",
        )
        self.assertEqual(len(factory.requests), 5)
        self.assertEqual(validate_run(self.data_root, run_id), [])
        self.assertEqual(replay_run(self.data_root, run_id), [])

    def test_metadata_and_infrastructure_failures_stop_and_preserve_unstarted(self) -> None:
        cases = {
            "ABSENT": [HTTPScenario(payload=models_payload(False))],
            "HTTP": [HTTPScenario(status=503, payload={})],
            "REDIRECT": [HTTPScenario(payload=models_payload()), HTTPScenario(status=302, payload={})],
            "MODEL": [HTTPScenario(payload=models_payload()), HTTPScenario(payload=completion_payload(0, model="wrong-model"))],
        }
        for suffix, scenarios in cases.items():
            with self.subTest(suffix=suffix):
                isolated = tempfile.TemporaryDirectory()
                try:
                    data_root = Path(isolated.name) / "data"
                    factory = FakeHTTPFactory(scenarios)
                    run_id, _ = execute_live_probe_with_configuration(
                        data_root,
                        self.config,
                        connection_factory=factory,
                        run_id=f"P1V2Q2C-PROBE-FAKE-{suffix}",
                    )
                    expected_calls = 1 if suffix in {"ABSENT", "HTTP"} else 2
                    self.assertEqual(len(factory.requests), expected_calls)
                    self.assertEqual(validate_run(data_root, run_id), [])
                    self.assertEqual(replay_run(data_root, run_id), [])
                finally:
                    isolated.cleanup()

    def test_total_deadline_prevents_connection_when_empty_and_after_metadata(self) -> None:
        clock = ManualClock()
        factory = FakeHTTPFactory(self.success_scenarios(), clock)
        transport = LiveHTTPTransport(self.config, factory)
        transcript = execute_protocol(
            transport,
            run_id="P1V2Q2C-PROBE-FAKE-ZERO",
            run_kind="live_probe",
            total_deadline_seconds=0.0,
            clock=clock,
        )
        self.assertEqual(factory.connection_calls, [])
        self.assertEqual(transcript["metadata"]["status"], "unstarted")
        clock = ManualClock()
        factory = FakeHTTPFactory([HTTPScenario(payload=models_payload(), duration_seconds=2.0)], clock)
        transport = LiveHTTPTransport(self.config, factory)
        transcript = execute_protocol(
            transport,
            run_id="P1V2Q2C-PROBE-FAKE-DEADLINE",
            run_kind="live_probe",
            total_deadline_seconds=1.0,
            clock=clock,
        )
        self.assertEqual(len(factory.connection_calls), 1)
        self.assertEqual(factory.connection_calls[0][2], 1.0)
        self.assertEqual(transcript["metadata"]["status"], "deadline_exhausted")
        self.assertTrue(all(not record["started"] for record in transcript["records"]))
        transport.clear_secret()

    def test_strict_configuration_and_one_time_reservation_fail_closed_before_http(self) -> None:
        invalids = (
            LiveConfiguration("http://localhost:58661/v1", LIVE_SECRET, TARGET_MODEL),
            LiveConfiguration(BASE_URL_IDENTITY + "/", LIVE_SECRET, TARGET_MODEL),
            LiveConfiguration(BASE_URL_IDENTITY + "?redirect=x", LIVE_SECRET, TARGET_MODEL),
            LiveConfiguration(BASE_URL_IDENTITY, LIVE_SECRET, "other-model"),
        )
        for configuration in invalids:
            factory = FakeHTTPFactory([])
            with self.assertRaises(ValueError):
                execute_live_probe_with_configuration(
                    self.data_root, configuration, connection_factory=factory
                )
            self.assertEqual(factory.connection_calls, [])
        first = reserve_live_attempt(self.data_root, "P1V2Q2C-PROBE-FAKE-RESERVED")
        self.assertTrue(first.is_file())
        with self.assertRaises(RuntimeError):
            reserve_live_attempt(self.data_root, "P1V2Q2C-PROBE-FAKE-SECOND")

    def test_cli_requires_flag_without_credential_loader_or_connection(self) -> None:
        script_path = CODE_ROOT / "scripts" / "run_q2c_probe.py"
        spec = importlib.util.spec_from_file_location("q2c_live_cli_test", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        calls: list[object] = []

        def fake_executor(data_root: Path, dotenv_path: Path) -> tuple[str, Path]:
            calls.append((data_root, dotenv_path))
            return "P1V2Q2C-PROBE-FAKE-CLI", data_root

        self.assertEqual(module.main([], executor=fake_executor), 2)
        self.assertEqual(calls, [])
        self.assertEqual(module.main(["--allow-live-q2c"], executor=fake_executor), 0)
        self.assertEqual(len(calls), 1)

    def test_live_evidence_tampering_fails_closed(self) -> None:
        factory = FakeHTTPFactory(self.success_scenarios())
        run_id, run_dir = execute_live_probe_with_configuration(
            self.data_root,
            self.config,
            connection_factory=factory,
            run_id="P1V2Q2C-PROBE-FAKE-TAMPER",
        )
        transcript_path = run_dir / "transcript.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript["records"][0]["request_sha256"] = "f" * 64
        transcript_path.write_text(canonical_json(transcript) + "\n", encoding="utf-8", newline="\n")
        self.assertTrue(validate_run(self.data_root, run_id))
        self.assertTrue(replay_run(self.data_root, run_id))

    def test_missing_delivery_is_reported_fail_closed_without_traceback(self) -> None:
        factory = FakeHTTPFactory(self.success_scenarios())
        run_id, run_dir = execute_live_probe_with_configuration(
            self.data_root,
            self.config,
            connection_factory=factory,
            run_id="P1V2Q2C-PROBE-FAKE-DELIVERY",
        )
        (run_dir / "reports" / "DELIVERY.md").unlink()
        self.assertIn("local_delivery_read", validate_run(self.data_root, run_id))
        self.assertTrue(replay_run(self.data_root, run_id))

    def test_full_execution_source_closure_is_bound_and_detects_drift(self) -> None:
        baseline = live_execution_source_hashes()
        self.assertIn("q2c_probe/config.py", baseline)
        self.assertIn("q2c_probe/evidence.py", baseline)
        self.assertIn("q2c_probe/runner.py", baseline)
        self.assertIn("scripts/run_q2c_probe.py", baseline)
        original_read_bytes = Path.read_bytes

        def altered_source(path: Path) -> bytes:
            value = original_read_bytes(path)
            if path.name == "config.py" and path.parent.name == "q2c_probe":
                return value + b"\n# simulated source drift\n"
            return value

        with patch.object(Path, "read_bytes", altered_source):
            self.assertNotEqual(live_execution_source_hashes(), baseline)

    def test_coordinated_run_kind_rechain_and_manifest_rewrite_is_rejected(self) -> None:
        factory = FakeHTTPFactory(self.success_scenarios())
        run_id, run_dir = execute_live_probe_with_configuration(
            self.data_root,
            self.config,
            connection_factory=factory,
            run_id="P1V2Q2C-PROBE-FAKE-KIND-TAMPER",
        )
        transcript_path = run_dir / "transcript.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        transcript["run_kind"] = "offline_fake"
        for key in (
            "live_policy_commitment",
            "live_runner_source_sha256",
            "live_transport_source_sha256",
            "metadata_call_count",
            "completion_call_count",
        ):
            transcript.pop(key, None)
        transcript_path.write_text(canonical_json(transcript) + "\n", encoding="utf-8", newline="\n")
        (run_dir / "ledger.json").write_text(
            canonical_json(build_ledger(transcript)) + "\n", encoding="utf-8", newline="\n"
        )
        semantic = transcript_semantic_view(transcript)
        rows = [
            {
                "profile": record["profile"],
                "started": record["started"],
                "outcome_category": record["outcome_category"],
                "expected_fields_satisfied": record["expected_fields_satisfied"],
            }
            for record in transcript["records"]
        ]
        tampered_outcome = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "run_kind": "offline_fake",
            "offline_status": "offline_fake_passed",
            "metadata_status": transcript["metadata"]["status"],
            "record_count": len(rows),
            "profile_outcomes": rows,
            "transcript_semantic_sha256": sha256_json(semantic),
        }
        tampered_report = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "report_kind": "offline_fake_transport_report",
            "offline_only": True,
            "metadata_status": tampered_outcome["metadata_status"],
            "record_count": tampered_outcome["record_count"],
            "profile_outcomes": rows,
            "replay_limit": "safe_summary_only",
        }
        (run_dir / "outcome.json").write_text(
            canonical_json(tampered_outcome) + "\n", encoding="utf-8", newline="\n"
        )
        (run_dir / "report.json").write_text(
            canonical_json(tampered_report) + "\n", encoding="utf-8", newline="\n"
        )
        (run_dir / "reports" / "DELIVERY.md").write_text(
            "# Q2-C Offline Fake Run Delivery\n\n- Run ID: `" + run_id + "`\n",
            encoding="utf-8",
            newline="\n",
        )
        (run_dir / "config_snapshot.json").write_text(
            canonical_json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "offline_only": True,
                    "profile_commitment": profile_commitment(),
                    "public_profile": public_profile_snapshot(),
                }
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        artifacts = {}
        for relative in REQUIRED_ARTIFACTS:
            digest, size = canonical_file_hash(run_dir / relative)
            artifacts[relative] = {"sha256": digest, "size_bytes": size}
        manifest_core = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "run_kind": "offline_fake",
            "profile_commitment": profile_commitment(),
            "artifacts": artifacts,
        }
        (run_dir / "run_manifest.json").write_text(
            canonical_json({**manifest_core, "manifest_core_sha256": sha256_json(manifest_core)}) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self.assertIn("run_id_kind_mismatch", validate_run(self.data_root, run_id))
        self.assertIn("replay_run_id_kind", replay_run(self.data_root, run_id))

    def test_coordinated_live_delivery_rewrite_with_no_network_claim_is_rejected(self) -> None:
        factory = FakeHTTPFactory(self.success_scenarios())
        run_id, run_dir = execute_live_probe_with_configuration(
            self.data_root,
            self.config,
            connection_factory=factory,
            run_id="P1V2Q2C-PROBE-FAKE-DELIVERY-RECHAIN",
        )
        transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
        # Simulate a coordinated local rewrite: recompute every ordinary chain
        # and manifest hash, but introduce a false offline/no-network claim.
        (run_dir / "ledger.json").write_text(
            canonical_json(build_ledger(transcript)) + "\n", encoding="utf-8", newline="\n"
        )
        outcome = derive_outcome(transcript)
        (run_dir / "outcome.json").write_text(
            canonical_json(outcome) + "\n", encoding="utf-8", newline="\n"
        )
        (run_dir / "report.json").write_text(
            canonical_json(derive_report(transcript, outcome)) + "\n", encoding="utf-8", newline="\n"
        )
        canonical_delivery = local_delivery_text(run_id, outcome)
        (run_dir / "reports" / "DELIVERY.md").write_text(
            canonical_delivery + "- Note: no network was used.\n",
            encoding="utf-8",
            newline="\n",
        )
        prior_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        core_keys = {"schema_version", "run_id", "run_kind", "profile_commitment", "artifacts", "manifest_core_sha256"}
        extra_core = {key: value for key, value in prior_manifest.items() if key not in core_keys}
        (run_dir / "run_manifest.json").write_text(
            canonical_json(artifact_manifest(run_dir, run_id, "live_probe", extra_core)) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self.assertIn(
            "local_delivery_delivery_forbidden_kind_claim",
            validate_run(self.data_root, run_id),
        )
        self.assertIn(
            "replay_delivery_forbidden_kind_claim",
            replay_run(self.data_root, run_id),
        )


if __name__ == "__main__":
    unittest.main()
