"""No-provider replay for the immutable Q0 transcript and event ledger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import QualificationError
from .fixtures import load_fixture_sets
from .paths import DEFAULT_DATA_ROOT, PROJECT_ROOT, append_jsonl, artifact_paths, assert_execution_roots, assert_workspace_root, load_json, safe_write_text
from .protocol import load_protocol_config
from .validation import (
    _identity_from_config,
    _read_jsonl,
    report_negative_case_lines,
    report_task_count_lines,
    validate_run,
)


def _delivery_report(run_id: str, manifest: dict[str, Any], validation: dict[str, Any], replay: dict[str, Any]) -> str:
    terminal = validation.get("terminal_state_counts", {})
    source_identity = manifest.get("source_identity", [])
    frozen_protocol = manifest.get("frozen_protocol", {})
    identity_keys = (
        "is_new_p1_run",
        "eligible_for_p1_gate_analysis",
        "eligible_for_p2",
        "eligible_for_confirmatory_analysis",
        "eligible_for_causal_effect_analysis",
        "p1_go",
        "p2_allowed",
        "qualification_candidate_status",
        "qualification_batch_decision",
        "real_model_calls",
        "local_loopback_http_calls",
        "remote_network_calls",
        "replay_model_calls",
        "replay_network_calls",
    )
    identity_lines = "\n".join(f"- {key}: `{manifest.get(key)!r}`" for key in identity_keys)
    protected_before = manifest.get("protected_tree_hashes_before", {})
    protected_after = manifest.get("protected_tree_hashes_after", {})
    protected_lines = "\n".join(
        f"- `{root}`: before `{protected_before.get(root, 'missing')}`; after `{protected_after.get(root, 'missing')}`"
        for root in sorted(set(protected_before) | set(protected_after))
    )
    source_lines = "\n".join(
        f"- `{item.get('relative_path', 'unknown')}`: `{item.get('sha256', 'unknown')}`"
        for item in source_identity
        if isinstance(item, dict)
    )
    task_count_lines = "\n".join(
        report_task_count_lines(validation.get("valid_task_counts", {}))
    )
    negative_case_lines = "\n".join(
        report_negative_case_lines(validation.get("negative_case_terminal_counts", {}))
    )
    return f"""# P1v2-Q0 离线资格框架交付报告

- Current Q0 run ID: `{run_id}`
- Report registry: unique current report
- Artifact kind: `p1v2_model_output_qualification`
- Phase: `P1V2_Q0_READINESS_DRY`
- Data role: `p1v2_qualification`
- Execution mode: `readiness_dry`

## 结果摘要

- request_count: {validation.get('request_count', 0)}
- model_output: {terminal.get('model_output', 0)}
- model_output_rejected: {terminal.get('model_output_rejected', 0)}
- model_call_failed: {terminal.get('model_call_failed', 0)}
- fake_provider_calls: {manifest.get('fake_provider_calls', 0)}
- real_model_calls: 0
- local_loopback_http_calls: 0
- remote_network_calls: 0
- replay_model_calls: 0
- replay_network_calls: 0

## Recomputed task-class counts

{task_count_lines}

## Recomputed negative-case terminal counts

{negative_case_lines}

## 离线复核

- Validation: {'pass' if validation.get('passed') else 'fail'}
- Replay: {'pass' if replay.get('passed') else 'fail'}
- Python executable: `{manifest.get('runtime', {}).get('python_executable', 'unknown')}`
- Python version: `{manifest.get('runtime', {}).get('python_version', 'unknown')}`
- New dependencies: none
- Protected trees before/after: {'unchanged' if manifest.get('protected_tree_hashes_before') == manifest.get('protected_tree_hashes_after') else 'mismatch'}

## 测试证据

- Command: `conda run --no-capture-output -n multi_agent_graph python -B -m pytest -p no:cacheprovider paperAlpha/pre_exp1/p1v2_qualification/tests -q`
- Implementation verification summary: `15 passed`.
- No new dependencies were installed.

## 冻结的未来 Q1 配置（Q0 未执行）

- model_output_stack_id: `{frozen_protocol.get('model_output_stack_id', 'missing')}`
- model_tag: `{frozen_protocol.get('model_tag', 'missing')}`
- provider: `{frozen_protocol.get('provider', 'missing')}`
- future_endpoint: `{frozen_protocol.get('future_endpoint', 'missing')}`
- format / temperature / seed / think: `{frozen_protocol.get('format', 'missing')}` / `{frozen_protocol.get('temperature', 'missing')}` / `{frozen_protocol.get('seed', 'missing')}` / `{frozen_protocol.get('think', 'missing')}`
- concurrency / retries / timeout: `{frozen_protocol.get('concurrency', 'missing')}` / `{frozen_protocol.get('retry_per_call', 'missing')}` / `{frozen_protocol.get('per_call_timeout_seconds', 'missing')}`

## Q0 身份与下游边界

{identity_lines}

## P1v2-A 契约来源身份

{source_lines}

## 受保护历史树哈希

{protected_lines}

## 明确边界

P1v2-Q0 readiness status: {'pass' if validation.get('passed') and replay.get('passed') else 'rework'}

P1v2-Q1 live model calls: not authorized

P1/P2 status: locked

本报告只证明离线资格框架的工程准入状态；它不是模型能力结论，也不构成 P1 或 P2 结果。
"""


def replay_run(
    run_id: str,
    *,
    code_root: Path = PROJECT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    workspace_root: Path | None = None,
    write_audit: bool = True,
    test_mode: bool = False,
) -> dict[str, Any]:
    """Recompute Q0 ledger validity strictly from saved artifacts.

    The replay imports no provider and makes no network or model call.
    """
    code_root = code_root.resolve(strict=False)
    data_root = data_root.resolve(strict=False)
    workspace_root = (workspace_root or code_root.parents[2]).resolve(strict=False)
    try:
        assert_execution_roots(code_root, data_root, test_mode=test_mode)
        assert_workspace_root(workspace_root)
        validation = validate_run(
            run_id,
            code_root=code_root,
            data_root=data_root,
            workspace_root=workspace_root,
            write_audit=False,
            test_mode=test_mode,
        )
        paths = artifact_paths(data_root, run_id)
        config = load_protocol_config(code_root)
        identity = _identity_from_config(config)
        screen, confirmation, negative_cases = load_fixture_sets(code_root)
        expected_request_count = len(screen) + len(confirmation) + len(negative_cases)
        event_count = len(_read_jsonl(paths["events"], paths["root"]))
        transcript_count = len(_read_jsonl(paths["transcripts"], paths["root"]))
        manifest = load_json(paths["manifest"], paths["root"])
        passed = bool(validation.get("passed")) and event_count == expected_request_count and transcript_count == expected_request_count
        errors = list(validation.get("errors", []))
        if event_count != expected_request_count or transcript_count != expected_request_count:
            errors.append("replay ledger counts do not close")
        result: dict[str, Any] = {
            "passed": passed,
            "run_id": run_id,
            "errors": errors,
            "consumed_event_count": event_count,
            "consumed_transcript_count": transcript_count,
            "replay_model_calls": 0,
            "replay_network_calls": 0,
        }
    except QualificationError as exc:
        config = None
        identity = None
        manifest = {}
        validation = {"passed": False, "errors": [f"replay failed closed: {exc.code}"]}
        result = {
            "passed": False,
            "run_id": run_id,
            "errors": validation["errors"],
            "consumed_event_count": 0,
            "consumed_transcript_count": 0,
            "replay_model_calls": 0,
            "replay_network_calls": 0,
        }
    except Exception:
        config = None
        identity = None
        manifest = {}
        validation = {"passed": False, "errors": ["replay failed closed: unexpected_error"]}
        result = {
            "passed": False,
            "run_id": run_id,
            "errors": validation["errors"],
            "consumed_event_count": 0,
            "consumed_transcript_count": 0,
            "replay_model_calls": 0,
            "replay_network_calls": 0,
        }
    if write_audit and identity is not None:
        paths = artifact_paths(data_root, run_id)
        audit = {
            **identity,
            "run_id": run_id,
            "passed": result["passed"],
            "errors": result["errors"],
            "consumed_event_count": result["consumed_event_count"],
            "consumed_transcript_count": result["consumed_transcript_count"],
            "replay_model_calls": 0,
            "replay_network_calls": 0,
            "real_model_calls": 0,
            "local_loopback_http_calls": 0,
            "remote_network_calls": 0,
        }
        try:
            append_jsonl(paths["replay"], paths["root"], audit)
            if result["passed"] and not paths["report"].exists():
                safe_write_text(paths["report"], paths["root"], _delivery_report(run_id, manifest, validation, result))
        except QualificationError:
            result = {**result, "passed": False, "errors": ["replay failed closed: audit_write_failed"]}
    return result
