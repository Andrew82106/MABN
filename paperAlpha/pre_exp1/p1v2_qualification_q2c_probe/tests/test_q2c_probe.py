from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


CODE_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from q2c_probe.config import CARDS, TARGET_MODEL, canonical_json, sha256_json
from q2c_probe.evidence import (
    REQUIRED_ARTIFACTS,
    artifact_manifest,
    build_ledger,
    derive_outcome,
    derive_report,
    local_delivery_text,
)
from q2c_probe.fake_scenarios import successful_completion_steps, successful_metadata_step
from q2c_probe.fake_transport import FakeStep, FakeTransport, ManualClock
from q2c_probe.historical_integrity import historical_tree_snapshot
from q2c_probe.replay import replay_run
from q2c_probe.runner import (
    classify_completion,
    create_offline_fake_run,
    execute_protocol,
    persist_offline_transcript,
)
from q2c_probe.validator import validate_run


def normal_envelope(index: int = 0) -> dict[str, object]:
    card = CARDS[index]
    content = card.expected_answer()
    if card.profile != "plain_text":
        content = json.dumps(
            {"probe_id": card.expected_probe_id(), "answer": card.expected_answer()},
            separators=(",", ":"),
            sort_keys=True,
        )
    return {
        "id": "test-request",
        "model": TARGET_MODEL,
        "system_fingerprint": "test-fingerprint",
        "provider_version": "test-provider",
        "choices": [{"message": {"content": content}}],
    }


def rewrite_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")


def rehash_manifest(run_dir: Path) -> None:
    manifest = artifact_manifest(
        run_dir,
        json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))["run_id"],
        "offline_fake",
    )
    rewrite_json(run_dir / "run_manifest.json", manifest)


class Q2CProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp.name) / "data"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_success_transport(self, clock: ManualClock | None = None) -> tuple[FakeTransport, ManualClock]:
        actual_clock = clock or ManualClock()
        return (
            FakeTransport(successful_metadata_step(), successful_completion_steps(), actual_clock),
            actual_clock,
        )

    def fake_run(self, suffix: str) -> tuple[str, Path]:
        return create_offline_fake_run(self.data_root, f"P1V2Q2C-FAKE-{suffix}")

    def assert_both_fail(self, run_id: str) -> None:
        self.assertTrue(validate_run(self.data_root, run_id))
        self.assertTrue(replay_run(self.data_root, run_id))

    def test_frozen_cards_and_all_success_forms(self) -> None:
        transport, clock = self.make_success_transport()
        transcript = execute_protocol(transport, run_id="P1V2Q2C-FAKE-SUCCESS", clock=clock)
        self.assertEqual([record["profile"] for record in transcript["records"]], [card.profile for card in CARDS])
        self.assertEqual([record["outcome_category"] for record in transcript["records"]], ["content_match"] * 4)
        self.assertEqual(len(transport.calls), 5)
        self.assertEqual(transcript["metadata"]["status"], "accepted")

    def test_response_classification_priority(self) -> None:
        card = CARDS[0]
        base = {
            "sequence": 1,
            "card_id": card.card_id,
            "profile": card.profile,
            "request_sha256": card.request_sha256,
            "started": True,
            "expected_fields_satisfied": False,
            "content_length": None,
            "content_sha256": None,
            "provider_identity": None,
            "timeout_seconds": 1.0,
            "elapsed_seconds": 0.0,
            "stop_reason": None,
        }
        scenarios = {
            "tool_calls": {"tool_calls": []},
            "function_call": {"function_call": {}},
            "refusal": {"refusal": "no"},
            "reasoning_detected": {"reasoning_content": "hidden"},
            "empty_content": {"content": ""},
        }
        for expected, message in scenarios.items():
            envelope = normal_envelope()
            envelope["choices"] = [{"message": message}]
            self.assertEqual(classify_completion(card, envelope, base)["outcome_category"], expected)
        self.assertEqual(classify_completion(card, {"model": TARGET_MODEL}, base)["outcome_category"], "malformed_envelope")
        mismatch = normal_envelope()
        mismatch["model"] = "other-model"
        self.assertEqual(classify_completion(card, mismatch, base)["outcome_category"], "model_mismatch")
        non_json = normal_envelope(1)
        non_json["choices"] = [{"message": {"content": "not-json"}}]
        self.assertEqual(classify_completion(CARDS[1], non_json, base)["outcome_category"], "non_json")
        wrong_fields = normal_envelope(1)
        wrong_fields["choices"] = [{"message": {"content": "{}"}}]
        self.assertEqual(classify_completion(CARDS[1], wrong_fields, base)["outcome_category"], "field_mismatch")

    def test_metadata_gate_and_transport_failures_stop(self) -> None:
        clock = ManualClock()
        missing = FakeTransport(FakeStep("response", {}), successful_completion_steps(), clock)
        transcript = execute_protocol(missing, run_id="P1V2Q2C-FAKE-META-MISSING", clock=clock)
        self.assertEqual(transcript["metadata"]["status"], "metadata_malformed")
        self.assertEqual(len(missing.calls), 1)
        self.assertTrue(all(not row["started"] for row in transcript["records"]))
        clock = ManualClock()
        absent = FakeTransport(
            FakeStep("response", {"data": [{"id": "different-model"}]}),
            successful_completion_steps(),
            clock,
        )
        transcript = execute_protocol(absent, run_id="P1V2Q2C-FAKE-META-ABSENT", clock=clock)
        self.assertEqual(transcript["metadata"]["status"], "metadata_target_absent")
        self.assertEqual(len(absent.calls), 1)
        clock = ManualClock()
        failed = FakeTransport(
            successful_metadata_step(),
            [FakeStep("error", "http_429")] + successful_completion_steps()[1:],
            clock,
        )
        transcript = execute_protocol(failed, run_id="P1V2Q2C-FAKE-HTTP", clock=clock)
        self.assertEqual(transcript["records"][0]["outcome_category"], "http_error")
        self.assertEqual(len(failed.calls), 2)
        self.assertTrue(all(not row["started"] for row in transcript["records"][1:]))

    def test_total_deadline_blocks_io_before_and_during_calls(self) -> None:
        clock = ManualClock()
        zero = FakeTransport(successful_metadata_step(), successful_completion_steps(), clock)
        transcript = execute_protocol(
            zero, run_id="P1V2Q2C-FAKE-ZERO", total_deadline_seconds=0.0, clock=clock
        )
        self.assertEqual(zero.calls, [])
        self.assertEqual(transcript["metadata"]["status"], "unstarted")
        clock = ManualClock()
        exhausted_metadata = FakeTransport(
            successful_metadata_step(duration_seconds=2.0), successful_completion_steps(), clock
        )
        transcript = execute_protocol(
            exhausted_metadata,
            run_id="P1V2Q2C-FAKE-META-DEADLINE",
            total_deadline_seconds=1.0,
            clock=clock,
        )
        self.assertEqual(len(exhausted_metadata.calls), 1)
        self.assertEqual(exhausted_metadata.calls[0][1], 1.0)
        self.assertEqual(transcript["metadata"]["status"], "deadline_exhausted")
        clock = ManualClock()
        during_completion = FakeTransport(
            successful_metadata_step(duration_seconds=0.1),
            [successful_completion_steps()[0], *successful_completion_steps()[1:]],
            clock,
        )
        during_completion.completion_steps[0] = FakeStep(
            "response", normal_envelope(0), duration_seconds=1.0
        )
        transcript = execute_protocol(
            during_completion,
            run_id="P1V2Q2C-FAKE-COMP-DEADLINE",
            total_deadline_seconds=0.5,
            clock=clock,
        )
        self.assertEqual(len(during_completion.calls), 2)
        self.assertEqual(transcript["records"][0]["outcome_category"], "deadline_exhausted")
        self.assertTrue(all(not row["started"] for row in transcript["records"][1:]))

    def test_transport_timeout_and_sensitive_raw_values_are_not_persisted(self) -> None:
        clock = ManualClock()
        timeout_transport = FakeTransport(
            successful_metadata_step(),
            [FakeStep("error", "timeout")] + successful_completion_steps()[1:],
            clock,
        )
        timeout_transcript = execute_protocol(
            timeout_transport, run_id="P1V2Q2C-FAKE-TIMEOUT", clock=clock
        )
        self.assertEqual(timeout_transcript["records"][0]["outcome_category"], "timeout")
        self.assertTrue(all(not row["started"] for row in timeout_transcript["records"][1:]))

        clock = ManualClock()
        raw_tool_transport = FakeTransport(
            successful_metadata_step(),
            [
                FakeStep(
                    "response",
                    {
                        "id": "safe-id",
                        "model": TARGET_MODEL,
                        "system_fingerprint": "safe-fingerprint",
                        "provider_version": "safe-version",
                        "choices": [
                            {
                                "message": {
                                    "content": "FAKE_SECRET",
                                    "tool_calls": [
                                        {"function": {"arguments": "FAKE_SECRET"}}
                                    ],
                                }
                            }
                        ],
                    },
                )
            ]
            + successful_completion_steps()[1:],
            clock,
        )
        transcript = execute_protocol(
            raw_tool_transport, run_id="P1V2Q2C-FAKE-REDACTION", clock=clock
        )
        self.assertEqual(transcript["records"][0]["outcome_category"], "tool_calls")
        before = historical_tree_snapshot()
        run_dir = persist_offline_transcript(self.data_root, transcript, before, historical_tree_snapshot())
        for relative in REQUIRED_ARTIFACTS:
            text = (run_dir / relative).read_text(encoding="utf-8")
            self.assertNotIn("FAKE_SECRET", text)
            self.assertNotIn("tool_arguments", text)

    def test_fake_run_validation_replay_and_artifact_redaction(self) -> None:
        run_id, run_dir = self.fake_run("VALID")
        self.assertEqual(validate_run(self.data_root, run_id), [])
        self.assertEqual(replay_run(self.data_root, run_id), [])
        self.assertEqual(validate_run(self.data_root, run_id), [])
        outcome = json.loads((run_dir / "outcome.json").read_text(encoding="utf-8"))
        delivery = (run_dir / "reports" / "DELIVERY.md").read_text(encoding="utf-8")
        self.assertEqual(delivery, local_delivery_text(run_id, outcome))
        for relative in REQUIRED_ARTIFACTS:
            text = (run_dir / relative).read_text(encoding="utf-8")
            self.assertNotIn("Q2C_OK", text)
            self.assertNotIn("FAKE_SECRET", text)
            self.assertNotIn("Authorization", text)
            self.assertNotIn("<think>", text)

    def test_individual_tampering_fails_closed_in_validator_and_replay(self) -> None:
        mutations = ("request", "identity", "category", "count", "ledger", "manifest")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                run_id, run_dir = self.fake_run("TAMPER-" + mutation.upper())
                if mutation == "request":
                    transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
                    transcript["records"][0]["request_sha256"] = "f" * 64
                    rewrite_json(run_dir / "transcript.json", transcript)
                elif mutation == "identity":
                    transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
                    transcript["records"][0]["provider_identity"]["request_id"] = "tampered-id"
                    rewrite_json(run_dir / "transcript.json", transcript)
                elif mutation == "category":
                    transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
                    transcript["records"][0]["outcome_category"] = "tool_calls"
                    rewrite_json(run_dir / "transcript.json", transcript)
                elif mutation == "count":
                    transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
                    transcript["records"].pop()
                    rewrite_json(run_dir / "transcript.json", transcript)
                elif mutation == "ledger":
                    ledger = json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))
                    ledger["events"][1]["payload"]["model_count"] = 99
                    rewrite_json(run_dir / "ledger.json", ledger)
                else:
                    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
                    manifest["artifacts"]["report.json"]["sha256"] = "0" * 64
                    rewrite_json(run_dir / "run_manifest.json", manifest)
                self.assert_both_fail(run_id)

    def test_coordinated_rechain_and_manifest_rehash_still_fails_fake_oracle(self) -> None:
        run_id, run_dir = self.fake_run("COORDINATED")
        transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
        transcript["records"][1]["outcome_category"] = "tool_calls"
        transcript["records"][1]["expected_fields_satisfied"] = False
        transcript["records"][1]["content_length"] = None
        transcript["records"][1]["content_sha256"] = None
        rewrite_json(run_dir / "transcript.json", transcript)
        rewrite_json(run_dir / "ledger.json", build_ledger(transcript))
        outcome = derive_outcome(transcript)
        rewrite_json(run_dir / "outcome.json", outcome)
        rewrite_json(run_dir / "report.json", derive_report(transcript, outcome))
        (run_dir / "reports" / "DELIVERY.md").write_text(
            local_delivery_text(run_id, outcome), encoding="utf-8", newline="\n"
        )
        rehash_manifest(run_dir)
        self.assert_both_fail(run_id)

    def test_coordinated_offline_delivery_rewrite_with_live_claim_is_rejected(self) -> None:
        run_id, run_dir = self.fake_run("DELIVERY-RECHAIN")
        transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
        # Rebuild all ordinary local links before inserting the contradictory
        # claim, so rejection cannot depend only on a stale manifest hash.
        rewrite_json(run_dir / "ledger.json", build_ledger(transcript))
        outcome = derive_outcome(transcript)
        rewrite_json(run_dir / "outcome.json", outcome)
        rewrite_json(run_dir / "report.json", derive_report(transcript, outcome))
        (run_dir / "reports" / "DELIVERY.md").write_text(
            local_delivery_text(run_id, outcome) + "- Note: live probe completed.\n",
            encoding="utf-8",
            newline="\n",
        )
        rehash_manifest(run_dir)
        self.assertIn(
            "local_delivery_delivery_forbidden_kind_claim",
            validate_run(self.data_root, run_id),
        )
        self.assertIn(
            "replay_delivery_forbidden_kind_claim",
            replay_run(self.data_root, run_id),
        )

    def test_run_local_delivery_isolation_and_historical_trees(self) -> None:
        before = historical_tree_snapshot()
        first_id, first_dir = self.fake_run("FIRST")
        first_delivery = (first_dir / "reports" / "DELIVERY.md").read_text(encoding="utf-8")
        second_id, _ = self.fake_run("SECOND")
        self.assertNotEqual(first_id, second_id)
        self.assertEqual((first_dir / "reports" / "DELIVERY.md").read_text(encoding="utf-8"), first_delivery)
        self.assertEqual(validate_run(self.data_root, first_id), [])
        self.assertEqual(replay_run(self.data_root, first_id), [])
        self.assertEqual(historical_tree_snapshot(), before)

    def test_path_safety_and_source_isolation(self) -> None:
        with self.assertRaises(ValueError):
            create_offline_fake_run(self.data_root, "../P1V2Q2C-FAKE-BAD")
        source_root = CODE_ROOT / "q2c_probe"
        source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.rglob("*.py"))
        for prohibited in ("import socket", "import urllib", "import requests", "import httpx", "os.environ"):
            self.assertNotIn(prohibited, source)
        self.assertNotIn("import p1v2_qualification_q2b", source)
        self.assertNotIn("import p1v2_qualification_q2b_repair", source)

    def test_runtime_socket_guard_and_no_env_read_for_validate_replay(self) -> None:
        transport, clock = self.make_success_transport()
        with patch("socket.socket", side_effect=AssertionError("socket_forbidden")):
            transcript = execute_protocol(transport, run_id="P1V2Q2C-FAKE-SOCKET", clock=clock)
        self.assertEqual(transcript["records"][0]["outcome_category"], "content_match")
        run_id, _ = self.fake_run("NOENV")
        original_read_text = Path.read_text

        def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
            if path.name == ".env":
                raise AssertionError("env_read_forbidden")
            return original_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", guarded_read_text):
            self.assertEqual(validate_run(self.data_root, run_id), [])
            self.assertEqual(replay_run(self.data_root, run_id), [])


if __name__ == "__main__":
    unittest.main()
