from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p1v2_qualification_q2a.common import DATA_ROOT, canonical_json, load_json, sha256_file, write_json
from p1v2_qualification_q2a.errors import MockTransportError, ProtocolViolation
from p1v2_qualification_q2a.protocol import adapt_openai_response, build_request, load_frozen_assets, validate_frozen_assets
from p1v2_qualification_q2a.replay import replay_run
from p1v2_qualification_q2a.runner import run_readiness
from p1v2_qualification_q2a.transport import MockOpenAITransport
from p1v2_qualification_q2a.validation import validate_run


LITERAL_MODEL = "gpt-5.3-codex-spark"
LITERAL_RUN_ID = "P1V2Q2A-READINESS-DRY-20260802T000000000000Z"
LITERAL_BASE = "http://127.0.0.1:58661/v1"


def _identity(request_id: str = "mock-test-identity", model: str = LITERAL_MODEL) -> dict[str, str]:
    return {
        "normalized_base_url_identity": LITERAL_BASE,
        "model_id": model,
        "provider_declared_model": model,
        "request_id": request_id,
        "provider_version": "not_provided",
    }


def _success_envelope(task: dict[str, object], serial: int = 1) -> str:
    content = {
        "agent_role": task["agent_role"],
        "episode_id": task["episode_id"],
        "decision": task["expected_decision"],
        "task_value": task["task_value"],
    }
    return canonical_json(
        {
            "choices": [{"message": {"content": canonical_json(content)}}],
            "model": LITERAL_MODEL,
            "provider_version": "not_provided",
            "x_request_id": "mock-test-" + str(serial),
        }
    )


def _mock_for(tasks: list[dict[str, object]], *, first: str | MockTransportError | None = None, identity_after: dict[str, str] | None = None) -> MockOpenAITransport:
    responses: list[str | MockTransportError] = [_success_envelope(task, index + 1) for index, task in enumerate(tasks)]
    if first is not None:
        responses[0] = first
    fake_key = "".join(("test", "_only", "_key"))
    return MockOpenAITransport(responses, fake_key, _identity(), identity_after or _identity("mock-test-after"))


class Q2AOfflineFrameworkTests(unittest.TestCase):
    def _temporary_root(self):
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=DATA_ROOT)

    def _run_success(self, temp_root: Path, *, asset_root: Path = ROOT) -> dict[str, object]:
        assets = load_frozen_assets(asset_root)
        tasks = assets.screen_tasks + assets.confirmation_tasks
        return run_readiness(
            transport=_mock_for(tasks),
            asset_root=asset_root,
            data_root=temp_root / "data",
            run_id=LITERAL_RUN_ID,
        )

    def _sync_artifact_hash(self, run_dir: Path, data_root: Path, relative: str) -> None:
        manifest_path = run_dir / "manifest.json"
        manifest = load_json(manifest_path)
        target = data_root / "reports" / "DELIVERY.md" if relative == "reports/DELIVERY.md" else run_dir / relative
        manifest["artifact_sha256"][relative] = sha256_file(target)
        write_json(manifest_path, manifest)

    def test_frozen_assets_and_one_profile_request(self) -> None:
        assets = load_frozen_assets(ROOT)
        validate_frozen_assets(assets)
        request = build_request(assets.screen_tasks[0], assets)
        self.assertEqual(set(request), {"model", "stream", "messages", "response_format"})
        self.assertEqual(request["model"], LITERAL_MODEL)
        self.assertIs(request["stream"], False)
        self.assertEqual(len(request["messages"]), 1)
        self.assertEqual(request["messages"][0]["role"], "user")
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertNotIn("temperature", request)
        self.assertNotIn("top_p", request)
        self.assertNotIn("max_tokens", request)
        self.assertNotIn("seed", request)

    def test_adapter_rejects_strict_envelope_and_content_errors(self) -> None:
        assets = load_frozen_assets(ROOT)
        task = assets.screen_tasks[0]
        valid = json.loads(_success_envelope(task))
        cases: list[tuple[str, str]] = []
        malformed_content = json.loads(_success_envelope(task))
        malformed_content["choices"][0]["message"]["content"] = "{"
        cases.append((canonical_json(malformed_content), "response_content_json"))
        fields = json.loads(_success_envelope(task))
        fields["choices"][0]["message"]["content"] = "{}"
        cases.append((canonical_json(fields), "response_schema"))
        for field, value, code in (
            ("agent_role", "publisher", "response_agent_role"),
            ("episode_id", "P1V2Q2A-SCREEN-999", "response_episode_id"),
            ("decision", "publish_safe", "response_decision"),
            ("task_value", "wrong", "response_task_value"),
        ):
            changed = json.loads(_success_envelope(task))
            content = json.loads(changed["choices"][0]["message"]["content"])
            content[field] = value
            changed["choices"][0]["message"]["content"] = canonical_json(content)
            cases.append((canonical_json(changed), code))
        empty = json.loads(_success_envelope(task))
        empty["choices"][0]["message"]["content"] = ""
        cases.append((canonical_json(empty), "empty_content"))
        non_string = json.loads(_success_envelope(task))
        non_string["choices"][0]["message"]["content"] = {"not": "a string"}
        cases.append((canonical_json(non_string), "empty_content"))
        cases.extend(
            [
                (canonical_json({"model": LITERAL_MODEL}), "missing_choices"),
                (canonical_json({"choices": [{}], "model": LITERAL_MODEL}), "unknown_envelope"),
                (canonical_json({"error": {"code": "mock"}, "model": LITERAL_MODEL}), "provider_error"),
                (canonical_json({"choices": [{"message": {"tool_calls": []}}], "model": LITERAL_MODEL}), "tool_calls"),
                (canonical_json({"choices": [{"message": {"refusal": "no"}}], "model": LITERAL_MODEL}), "refusal"),
                (canonical_json({"choices": [{"message": {"reasoning_content": "hidden", "content": "{}"}}], "model": LITERAL_MODEL}), "reasoning_detected"),
                (canonical_json({"choices": [{"message": {"reasoning": "hidden", "content": "{}"}}], "model": LITERAL_MODEL}), "reasoning_detected"),
                (canonical_json({"choices": [{"message": {"content": "<think>x</think>"}}], "model": LITERAL_MODEL}), "reasoning_detected"),
                (canonical_json({"choices": [{"message": {"content": "{}"}}], "model": "different-model"}), "model_mismatch"),
            ]
        )
        for raw, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(ProtocolViolation) as raised:
                    adapt_openai_response(raw, task, assets)
                self.assertEqual(raised.exception.code, expected_code)
        with self.assertRaises(ProtocolViolation) as raised:
            adapt_openai_response("not-json", task, assets)
        self.assertEqual(raised.exception.code, "response_envelope_json")
        adapted = adapt_openai_response(canonical_json(valid), task, assets)
        self.assertEqual(adapted.parsed["episode_id"], task["episode_id"])

    def test_mock_failure_categories_are_deterministic_and_offline(self) -> None:
        categories = [
            "mock_http_401",
            "mock_http_403",
            "mock_http_429",
            "mock_http_500",
            "mock_timeout",
            "mock_redirect",
            "mock_tls_error",
            "mock_dns_error",
        ]
        for category in categories:
            with self.subTest(category=category):
                mock = MockOpenAITransport([MockTransportError(category)], "".join(("key", "_for", "_test")), _identity())
                with self.assertRaises(MockTransportError) as raised:
                    mock.chat({})
                self.assertEqual(raised.exception.category, category)
                self.assertEqual(mock.counts["real_network_calls"], 0)
                self.assertEqual(mock.counts["real_model_calls"], 0)
                self.assertEqual(mock.counts["real_credential_reads"], 0)

    def test_success_flow_validates_and_replays(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            result = self._run_success(root)
            self.assertEqual(result["decision"], "ready_for_q2b")
            self.assertEqual(result["call_counts"]["real_network_calls"], 0)
            self.assertEqual(result["call_counts"]["real_model_calls"], 0)
            checked = validate_run(LITERAL_RUN_ID, data_root=root / "data")
            replayed = replay_run(LITERAL_RUN_ID, data_root=root / "data")
            self.assertTrue(checked["passed"], checked)
            self.assertTrue(replayed["passed"], replayed)

    def test_screen_failure_blocks_all_confirmation_cards(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            assets = load_frozen_assets(ROOT)
            tasks = assets.screen_tasks + assets.confirmation_tasks
            result = run_readiness(
                transport=_mock_for(tasks, first=MockTransportError("mock_http_429")),
                data_root=root / "data",
                run_id=LITERAL_RUN_ID,
            )
            transcript = load_json(Path(result["run_dir"]) / "transcript.json")
            confirmations = [record for record in transcript["records"] if record["phase"] == "confirmation"]
            self.assertEqual(result["decision"], "rework")
            self.assertFalse(transcript["confirmation_started"])
            self.assertEqual(len(confirmations), 96)
            self.assertTrue(all(record["status"] == "unstarted" for record in confirmations))
            self.assertTrue(validate_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])
            self.assertTrue(replay_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])

    def test_identity_profile_drift_produces_rework_without_real_io(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            assets = load_frozen_assets(ROOT)
            tasks = assets.screen_tasks + assets.confirmation_tasks
            result = run_readiness(
                transport=_mock_for(tasks, identity_after=_identity("drift", "different-model")),
                data_root=root / "data",
                run_id=LITERAL_RUN_ID,
            )
            self.assertEqual(result["decision"], "rework")
            self.assertEqual(result["call_counts"]["real_network_calls"], 0)
            self.assertTrue(validate_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])
            self.assertTrue(replay_run(LITERAL_RUN_ID, data_root=root / "data")["passed"])

    def test_fake_key_and_authorization_never_reach_artifacts(self) -> None:
        with self._temporary_root() as temporary:
            root = Path(temporary)
            assets = load_frozen_assets(ROOT)
            tasks = assets.screen_tasks + assets.confirmation_tasks
            marker = "".join(("isolated", "_fake", "_credential"))
            result = run_readiness(
                transport=MockOpenAITransport([_success_envelope(task, index + 1) for index, task in enumerate(tasks)], marker, _identity(), _identity("after")),
                data_root=root / "data",
                run_id=LITERAL_RUN_ID,
            )
            artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in Path(result["run_dir"]).rglob("*") if path.is_file())
            artifact_text += (root / "data" / "reports" / "DELIVERY.md").read_text(encoding="utf-8")
            self.assertNotIn(marker, artifact_text)
            self.assertNotIn("authorization", artifact_text.lower())

    def test_validator_and_replay_are_isolated_from_execution_module(self) -> None:
        validation_source = (ROOT / "src" / "p1v2_qualification_q2a" / "validation.py").read_text(encoding="utf-8")
        replay_source = (ROOT / "src" / "p1v2_qualification_q2a" / "replay.py").read_text(encoding="utf-8")
        initializer = (ROOT / "src" / "p1v2_qualification_q2a" / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn(".transport", validation_source)
        self.assertNotIn(".transport", replay_source)
        self.assertNotIn("from .runner", initializer)

    def test_frozen_assets_and_source_contracts_are_tamper_detected(self) -> None:
        targets = [
            "configs/q2a_remote_protocol.json",
            "configs/q2a_provider_profile.json",
            "prompts/response_contract.txt",
            "schemas/response_contract.schema.json",
            "fixtures/screen_tasks.json",
            "fixtures/confirmation_tasks.json",
            "permissions/public_sink_rule.json",
            "provenance/source_contracts.json",
        ]
        with self._temporary_root() as temporary:
            root = Path(temporary)
            self._run_success(root)
            for index, relative in enumerate(targets):
                with self.subTest(relative=relative):
                    asset_copy = root / ("assets_" + str(index))
                    shutil.copytree(ROOT, asset_copy, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
                    target = asset_copy / relative
                    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                    checked = validate_run(LITERAL_RUN_ID, asset_root=asset_copy, data_root=root / "data")
                    self.assertFalse(checked["passed"])

    def test_artifact_count_run_id_manifest_and_report_tampering_is_detected(self) -> None:
        cases = ("ledger.jsonl", "transcript.json", "outcome.json", "readiness_report.json", "reports/DELIVERY.md", "manifest.json", "run_id")
        for case in cases:
            with self.subTest(case=case), self._temporary_root() as temporary:
                root = Path(temporary)
                result = self._run_success(root)
                data_root = root / "data"
                run_dir = Path(result["run_dir"])
                if case == "ledger.jsonl":
                    with (run_dir / case).open("a", encoding="utf-8") as handle:
                        handle.write("{}\n")
                    self._sync_artifact_hash(run_dir, data_root, case)
                elif case == "transcript.json":
                    transcript = load_json(run_dir / case)
                    transcript["records"][0]["response"]["task_value"] = "tampered"
                    write_json(run_dir / case, transcript)
                    self._sync_artifact_hash(run_dir, data_root, case)
                elif case == "outcome.json":
                    outcome = load_json(run_dir / case)
                    outcome["call_counts"]["real_network_calls"] = 1
                    write_json(run_dir / case, outcome)
                    self._sync_artifact_hash(run_dir, data_root, case)
                elif case == "readiness_report.json":
                    report = load_json(run_dir / case)
                    report["real_model_call_count"] = 1
                    write_json(run_dir / case, report)
                    self._sync_artifact_hash(run_dir, data_root, case)
                elif case == "reports/DELIVERY.md":
                    delivery = data_root / "reports" / "DELIVERY.md"
                    delivery.write_text(delivery.read_text(encoding="utf-8").replace("Real network calls: 0", "Real network calls: 1"), encoding="utf-8")
                    self._sync_artifact_hash(run_dir, data_root, case)
                elif case == "manifest.json":
                    manifest = load_json(run_dir / case)
                    manifest["decision"] = "rework"
                    write_json(run_dir / case, manifest)
                else:
                    outcome = load_json(run_dir / "outcome.json")
                    outcome["run_id"] = "P1V2Q2A-READINESS-DRY-20260802T000000000001Z"
                    write_json(run_dir / "outcome.json", outcome)
                    self._sync_artifact_hash(run_dir, data_root, "outcome.json")
                checked = validate_run(LITERAL_RUN_ID, data_root=data_root)
                self.assertFalse(checked["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
