"""Pure admission workflow and an in-memory public sink used only in tests/dry runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contract import ContractPolicy, SchemaContract, public_report_hash, validate_response


@dataclass
class MockPublicSink:
    published_reports: list[dict[str, Any]] = field(default_factory=list)

    def publish(self, report: dict[str, Any]) -> None:
        self.published_reports.append(dict(report))


def evaluate_response(
    *,
    raw_response: Any,
    expected_role: Any,
    expected_episode_id: Any,
    policy: ContractPolicy,
    schema: SchemaContract | None = None,
    sink: MockPublicSink | None = None,
) -> dict[str, Any]:
    result = validate_response(raw_response, expected_role, expected_episode_id, policy, schema=schema)
    should_publish = result["contract_valid"] is True and result["decision"] == policy.allow_continue_value
    if should_publish and sink is not None:
        report = result["public_report"]
        if not isinstance(report, dict):
            should_publish = False
        else:
            sink.publish(report)
    return {
        "contract_valid": result["contract_valid"],
        "decision": result["decision"],
        "public_sink_published": should_publish,
        "public_report_sha256": public_report_hash(result["public_report"] if should_publish else None),
        "rejection_code": result["rejection_code"],
    }
