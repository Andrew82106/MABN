"""Graph-driven scripted P0 sandbox and four-trajectory smoke runner."""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .event_log import AppendOnlyEventLog, utc_now
from .gate import MessageGate
from .mock_tools import LocalMockTools
from .models import Message, Outcome, RunManifest, TreatmentName
from .paths import (
    DEFAULT_PATHS,
    PAPER_ALPHA_ROOT,
    ProjectPaths,
    ensure_data_dirs,
    relative_to_paper_alpha,
    require_data_path,
)
from .permissions import PermissionEnforcer
from .provenance import (
    capture_python_environment,
    relative_hashes,
    source_file_hashes,
    workspace_state,
)
from .runtime_config import RuntimeConfig
from .state import apply_replay_action, build_initial_state, stable_hash


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any, paths: ProjectPaths) -> None:
    output = require_data_path(path, paths.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_jsonl_new(
    path: Path,
    rows: list[dict[str, Any]],
    paths: ProjectPaths,
) -> None:
    output = require_data_path(path, paths.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def _write_text(path: Path, text: str, paths: ProjectPaths) -> None:
    output = require_data_path(path, paths.data_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _transition(
    *,
    state: dict[str, Any],
    action: dict[str, Any],
    event_log: AppendOnlyEventLog,
    episode_id: str,
    event_type: str,
    agent_id: str | None,
    edge_id: str | None,
    treatment: str,
    payload: Any,
) -> None:
    before = stable_hash(state)
    apply_replay_action(state, action)
    after = stable_hash(state)
    event_log.append(
        episode_id=episode_id,
        event_type=event_type,
        agent_id=agent_id,
        edge_id=edge_id,
        treatment=treatment,
        payload=payload,
        state_before_hash=before,
        state_after_hash=after,
        details={"replay_action": action},
    )


def _episode_secret(state: dict[str, Any]) -> str:
    record = next(iter(state["vendor_records"].values()))
    return record["test_secret"]


def _deliver_along_execution_order(
    *,
    runtime: RuntimeConfig,
    gate: MessageGate,
    state: dict[str, Any],
    event_log: AppendOnlyEventLog,
    episode_id: str,
    treatment: TreatmentName,
    content: dict[str, Any],
    start_index: int,
    end_index: int,
    message_label: str,
) -> dict[str, Any]:
    delivered_content = copy.deepcopy(content)
    order = runtime.graph.execution_order
    for edge_index in range(start_index, end_index):
        source = order[edge_index]
        target = order[edge_index + 1]
        edge = runtime.graph.edge_between(source, target)
        message = Message(
            message_id=(
                f"MSG-{episode_id}-{message_label}-{edge_index + 1:02d}"
            ),
            sender_id=source,
            recipient_id=target,
            edge_id=edge.edge_id,
            content=copy.deepcopy(delivered_content),
        )
        runtime.graph.validate_message(message)
        gate_result = gate.apply(
            message,
            treatment,
            event_log=event_log,
            episode_id=episode_id,
            state_hash=stable_hash(state),
        )
        delivered = gate_result.message
        _transition(
            state=state,
            action={"type": "record_message", "message": delivered.to_dict()},
            event_log=event_log,
            episode_id=episode_id,
            event_type="message_delivered",
            agent_id=delivered.recipient_id,
            edge_id=delivered.edge_id,
            treatment=treatment,
            payload=delivered.content,
        )
        delivered_content = copy.deepcopy(delivered.content)
    return delivered_content


def run_scripted_episode(
    *,
    run_id: str,
    episode_id: str,
    scenario: str,
    treatment: TreatmentName,
    event_log: AppendOnlyEventLog,
    runtime: RuntimeConfig | None = None,
) -> tuple[dict[str, Any], Outcome]:
    runtime = runtime or RuntimeConfig.load()
    condition = runtime.condition(scenario)
    if treatment != condition.treatment:
        raise ValueError(
            f"Scenario {scenario!r} requires treatment "
            f"{condition.treatment!r}, got {treatment!r}"
        )
    fixture = _load_json(runtime.vendor_fixture_path)["records"][0]
    input_template = runtime.templates.get(condition.input_template_id)
    state = build_initial_state(episode_id, fixture)
    initial_hash = stable_hash(state)
    event_log.append(
        episode_id=episode_id,
        event_type="episode_started",
        agent_id=None,
        edge_id=None,
        treatment=treatment,
        payload={"scenario": scenario},
        state_before_hash=initial_hash,
        state_after_hash=initial_hash,
        details={
            "scenario": scenario,
            "input_template_id": input_template.template_id,
            "input_template_version": input_template.version,
            "input_template_hash": input_template.content_hash,
            "initial_state": copy.deepcopy(state),
        },
    )
    gate = MessageGate.from_runtime(
        runtime.graph,
        runtime.treatments,
        runtime.templates,
    )
    permissions = PermissionEnforcer.from_config(runtime.agents_path)
    tools = LocalMockTools(
        state=state,
        permissions=permissions,
        event_log=event_log,
        run_id=run_id,
        episode_id=episode_id,
        treatment=treatment,
    )
    order = runtime.graph.execution_order
    reader_index = order.index(runtime.reader_agent_id)
    publisher_index = order.index(runtime.publisher_agent_id)
    request = _deliver_along_execution_order(
        runtime=runtime,
        gate=gate,
        state=state,
        event_log=event_log,
        episode_id=episode_id,
        treatment=treatment,
        content=input_template.content,
        start_index=0,
        end_index=reader_index,
        message_label="REQUEST",
    )
    if request.get("noop") is True:
        _transition(
            state=state,
            action={
                "type": "set_task_status",
                "status": "incomplete_information",
            },
            event_log=event_log,
            episode_id=episode_id,
            event_type="task_status_changed",
            agent_id=runtime.reader_agent_id,
            edge_id=None,
            treatment=treatment,
            payload={"status": "incomplete_information"},
        )
    else:
        record = tools.get_internal_vendor_record(
            request["vendor_id"],
            agent_id=runtime.reader_agent_id,
            idempotency_key=f"{episode_id}:read:vendor",
        )
        report = {
            field: copy.deepcopy(record[field])
            for field in request["requested_fields"]
            if field in record
        }
        delivered_report = _deliver_along_execution_order(
            runtime=runtime,
            gate=gate,
            state=state,
            event_log=event_log,
            episode_id=episode_id,
            treatment=treatment,
            content=report,
            start_index=reader_index,
            end_index=publisher_index,
            message_label="REPORT",
        )
        tools.publish_public_report(
            delivered_report,
            agent_id=runtime.publisher_agent_id,
            idempotency_key=f"{episode_id}:publish:report",
        )
        _transition(
            state=state,
            action={"type": "set_task_status", "status": "completed"},
            event_log=event_log,
            episode_id=episode_id,
            event_type="task_status_changed",
            agent_id=runtime.publisher_agent_id,
            edge_id=None,
            treatment=treatment,
            payload={"status": "completed"},
        )
    fake_secret = _episode_secret(state)
    publications_text = json.dumps(
        state["tool_state"]["publications"],
        ensure_ascii=False,
        sort_keys=True,
    )
    outcome_body = {
        "scenario": scenario,
        "treatment": treatment,
        "leak_detected": fake_secret in publications_text,
        "task_success": state["task_status"] == "completed",
        "task_status": state["task_status"],
    }
    _transition(
        state=state,
        action={"type": "set_outcome", "outcome": outcome_body},
        event_log=event_log,
        episode_id=episode_id,
        event_type="outcome_recorded",
        agent_id=None,
        edge_id=None,
        treatment=treatment,
        payload=outcome_body,
    )
    final_hash = stable_hash(state)
    outcome = Outcome(
        run_id=run_id,
        episode_id=episode_id,
        scenario=scenario,
        treatment=treatment,
        leak_detected=outcome_body["leak_detected"],
        task_success=outcome_body["task_success"],
        task_status=outcome_body["task_status"],
        final_state_hash=final_hash,
    )
    event_log.append(
        episode_id=episode_id,
        event_type="episode_completed",
        agent_id=None,
        edge_id=None,
        treatment=treatment,
        payload=outcome.to_dict(),
        state_before_hash=final_hash,
        state_after_hash=final_hash,
        details={
            "scenario": scenario,
            "treatment": treatment,
            "task_status": outcome.task_status,
            "final_state_hash": final_hash,
        },
    )
    return state, outcome


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"P0-SMOKE-{stamp}"


def _validate_run_id(run_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) is None:
        raise ValueError("run_id may contain only letters, digits, dot, underscore, dash")


def _smoke_report(
    run_id: str,
    outcomes: list[Outcome],
    replay_passed: bool,
    behavior_passed: bool,
    python_environment: dict[str, Any],
) -> str:
    lines = [
        f"# P0 smoke-test 摘要：{run_id}",
        "",
        "本报告只验证确定性基础设施行为，不代表实验性研究结论。",
        "",
        f"- sys.executable：`{python_environment['sys_executable']}`",
        f"- Python：`{python_environment['python_version']}`",
        f"- pytest：`{python_environment['pytest_version']}`",
        "",
        "| scripted 轨迹 | treatment | 泄漏记录 | 任务成功 | 状态 |",
        "|---|---|---:|---:|---|",
    ]
    for outcome in outcomes:
        lines.append(
            "| {scenario} | {treatment} | {leak} | {success} | {status} |".format(
                scenario=outcome.scenario,
                treatment=outcome.treatment,
                leak=str(outcome.leak_detected).lower(),
                success=str(outcome.task_success).lower(),
                status=outcome.task_status,
            )
        )
    lines.extend(
        [
            "",
            f"- 预期行为检查：{'PASS' if behavior_passed else 'FAIL'}",
            f"- transcript replay：{'PASS' if replay_passed else 'FAIL'}",
            "- replay 执行模式：pure_state_actions",
            "- replay 实时工具调用：0",
            "- 外部网络或真实系统调用：0",
            "",
        ]
    )
    return "\n".join(lines)


def run_p0_smoke(
    run_id: str | None = None,
    *,
    project_root: Path = PAPER_ALPHA_ROOT,
) -> dict[str, Any]:
    """Run exactly four graph-driven deterministic scripted trajectories."""

    from .replay import replay_run
    from .validation import validate_run

    paths = (
        DEFAULT_PATHS
        if project_root.resolve() == PAPER_ALPHA_ROOT.resolve()
        else ProjectPaths.from_root(project_root)
    )
    ensure_data_dirs(paths)
    runtime = RuntimeConfig.load(paths)
    python_environment = capture_python_environment(enforce=True)
    run_id = run_id or _default_run_id()
    _validate_run_id(run_id)
    event_path = paths.raw_dir / f"events_{run_id}.jsonl"
    manifest_path = paths.manifests_dir / f"manifest_{run_id}.json"
    outcomes_path = paths.processed_dir / f"outcomes_{run_id}.jsonl"
    replay_path = paths.interim_dir / f"replay_{run_id}.json"
    smoke_report_path = paths.reports_dir / f"smoke_{run_id}.md"
    validation_json_path = paths.interim_dir / f"validation_{run_id}.json"
    validation_report_path = paths.reports_dir / f"validation_{run_id}.md"
    output_paths = (
        event_path,
        manifest_path,
        outcomes_path,
        replay_path,
        smoke_report_path,
        validation_json_path,
        validation_report_path,
    )
    for path in output_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing run output: {path}")
    output_files = [
        relative_to_paper_alpha(path, paths.paper_alpha_root)
        for path in output_paths
    ]
    manifest = RunManifest(
        run_id=run_id,
        experiment_stage="p0_infrastructure",
        config_hashes=relative_hashes(
            runtime.config_paths,
            paths.paper_alpha_root,
        ),
        shared_input_hashes=relative_hashes(
            runtime.shared_input_paths,
            paths.paper_alpha_root,
        ),
        message_template_hashes=runtime.templates.hashes(),
        source_file_hashes=source_file_hashes(paths),
        workspace_state=workspace_state(paths),
        python_environment=python_environment,
        random_seed=int(runtime.experiment["seed"]),
        started_at=utc_now(),
        ended_at=None,
        status="running",
        output_files=output_files,
        validation={},
    )
    _write_json(manifest_path, manifest.to_dict(), paths)
    _write_json(
        validation_json_path,
        {
            "run_id": run_id,
            "passed": False,
            "checks": {},
            "errors": ["validation_pending"],
            "event_count": 0,
            "episode_count": 0,
        },
        paths,
    )
    _write_text(
        validation_report_path,
        f"# 运行验证摘要：{run_id}\n\n总体结果：PENDING\n",
        paths,
    )
    event_log = AppendOnlyEventLog(
        event_path,
        run_id,
        data_root=paths.data_root,
    )
    outcomes: list[Outcome] = []
    try:
        for index, condition in enumerate(runtime.conditions, start=1):
            episode_id = f"{run_id}-EP{index:02d}"
            _, outcome = run_scripted_episode(
                run_id=run_id,
                episode_id=episode_id,
                scenario=condition.scenario,
                treatment=condition.treatment,
                event_log=event_log,
                runtime=runtime,
            )
            outcomes.append(outcome)
        behavior_passed = all(
            {
                "leak_detected": outcome.leak_detected,
                "task_success": outcome.task_success,
                "task_status": outcome.task_status,
            }
            == runtime.condition(outcome.scenario).expected
            for outcome in outcomes
        )
        if not behavior_passed:
            raise AssertionError("One or more smoke trajectories violated P0 expectations")
        _write_jsonl_new(
            outcomes_path,
            [outcome.to_dict() for outcome in outcomes],
            paths,
        )
        replay_summary = replay_run(
            run_id,
            write_output=True,
            paths=paths,
        )
        if not replay_summary["passed"]:
            raise AssertionError("Transcript replay hash mismatch")
        _write_text(
            smoke_report_path,
            _smoke_report(
                run_id,
                outcomes,
                replay_summary["passed"],
                behavior_passed,
                python_environment,
            ),
            paths,
        )
        manifest.ended_at = utc_now()
        manifest.status = "completed"
        manifest.validation = {
            "smoke_expected_behavior": behavior_passed,
            "replay_hashes_match": replay_summary["passed"],
            "replay_live_tool_calls": replay_summary["live_tool_calls"],
            "data_schema_validation": False,
            "validation_checks": {},
        }
        _write_json(manifest_path, manifest.to_dict(), paths)
        validation_summary = validate_run(
            run_id,
            update_manifest=True,
            write_outputs=True,
            paths=paths,
        )
        if not validation_summary["passed"]:
            raise AssertionError(
                "Run validation failed: "
                + "; ".join(validation_summary["errors"])
            )
        return {
            "run_id": run_id,
            "passed": True,
            "behavior_passed": behavior_passed,
            "replay_passed": replay_summary["passed"],
            "validation_passed": validation_summary["passed"],
            "python_environment": python_environment,
            "outcomes": [outcome.to_dict() for outcome in outcomes],
            "output_files": output_files,
        }
    except Exception:
        manifest.ended_at = utc_now()
        manifest.status = "failed"
        _write_json(manifest_path, manifest.to_dict(), paths)
        raise
