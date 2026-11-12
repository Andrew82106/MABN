'Validate and report the checked-in A2ASecBench v1 run artifacts.\n\nThe validator is intentionally read-only with respect to benchmark inputs and\nrun JSONL.  It treats the five subdirectories under ``experiments/a2asecbench/runs/a2asecbench_v1``\nas the canonical outputs and writes only a manifest/report beside them.  A\nbenchmark ``success`` field is used for attack ASR; controls are scored from\nthe explicit ``details.control_pass`` field, never from the attack ``success``\nfield (which is false by construction for controls).\n'

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'experiments/a2asecbench/runs/a2asecbench_v1'

RUN_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "as", "attack_id": "AS", "config": 'knowledge/templates/offline/as.yaml', "attack_cases": 100, "control_cases": 0},
    {"name": "asrf_eval", "attack_id": "ASRF", "config": 'knowledge/templates/offline/asrf_eval.yaml', "attack_cases": 100, "control_cases": 100},
    {"name": "atsi_eval", "attack_id": "ATSI", "config": 'knowledge/templates/offline/atsi_eval.yaml', "attack_cases": 100, "control_cases": 100},
    {"name": "co_eval", "attack_id": "CO", "config": 'knowledge/templates/offline/co_eval.yaml', "attack_cases": 100, "control_cases": 100},
    {"name": "hotf_eval", "attack_id": "HOTF", "config": 'knowledge/templates/offline/hotf_eval.yaml', "attack_cases": 100, "control_cases": 100},
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Return valid object rows, invalid-line records, and blank-line count."""
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    blanks = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        return rows, [{"line": 0, "error": f"{type(exc).__name__}: {exc}"}], blanks
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            blanks += 1
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            invalid.append({"line": line_no, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not isinstance(value, dict):
            invalid.append({"line": line_no, "error": f"expected object, got {type(value).__name__}"})
            continue
        rows.append(value)
    return rows, invalid, blanks


def _row_issues(row: dict[str, Any], line_no: int) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    required = ("attack_id", "case_id", "details", "success", "timestamp", "trial_index")
    for key in required:
        if key not in row:
            issues.append({"line": line_no, "field": key, "error": "missing"})
    if not isinstance(row.get("details"), dict):
        issues.append({"line": line_no, "field": "details", "error": "expected object"})
        return issues
    details = row["details"]
    if not isinstance(details.get("is_control", False), bool):
        issues.append({"line": line_no, "field": "details.is_control", "error": "expected boolean"})
    events = details.get("events", [])
    if events is not None and not isinstance(events, list):
        issues.append({"line": line_no, "field": "details.events", "error": "expected array or null"})
    elif isinstance(events, list):
        for event_index, event in enumerate(events):
            if not isinstance(event, dict):
                issues.append({"line": line_no, "field": f"details.events[{event_index}]", "error": "expected object"})
                continue
            if not (event.get("event_type") or event.get("type")):
                issues.append({"line": line_no, "field": f"details.events[{event_index}]", "error": "missing event_type/type"})
    return issues


def _canonical_execution(name: str, directory: Path, base: Path) -> dict[str, Any]:
    """Describe the command/log files that produced the canonical run."""
    if name == "co_eval" and (directory / "isolated_manifest.json").is_file():
        manifest, manifest_error = _read_json(directory / "isolated_manifest.json")
        exit_path = directory / "isolated_runner_exit_code.txt"
        command_path = directory / "isolated_command.txt"
        stdout_path = directory / "isolated_stdout.log"
        stderr_path = directory / "isolated_stderr.log"
        return {
            "mode": "isolated_per_case_recovery",
            "command": _relative(command_path, base),
            "stdout": _relative(stdout_path, base),
            "stderr": _relative(stderr_path, base),
            "exit_code_file": _relative(exit_path, base),
            "exit_code": _text_int(exit_path),
            "manifest": _relative(directory / "isolated_manifest.json", base),
            "manifest_valid": manifest.get("valid") if isinstance(manifest, dict) else False,
            "manifest_error": manifest_error,
            "processed_cases": manifest.get("processed_cases") if isinstance(manifest, dict) else None,
            "expected_cases": manifest.get("expected_cases") if isinstance(manifest, dict) else None,
            "timeout_s": manifest.get("timeout_s") if isinstance(manifest, dict) else None,
            "original_attempt": {
                "command": _relative(directory / "command.txt", base),
                "stdout": _relative(directory / "stdout.txt", base),
                "stderr": _relative(directory / "stderr.txt", base),
                "exit_code_file": _relative(directory / "exit_code.txt", base),
                "exit_code": _text_int(directory / "exit_code.txt"),
                "included_in_canonical": False,
                "reason": "initial aggregate orchestration stopped before completion; replaced by isolated recovery",
            },
        }
    return {
        "mode": "single_orchestration",
        "command": _relative(directory / "command.txt", base),
        "stdout": _relative(directory / "stdout.txt", base),
        "stderr": _relative(directory / "stderr.txt", base),
        "exit_code_file": _relative(directory / "exit_code.txt", base),
        "exit_code": _text_int(directory / "exit_code.txt"),
    }


def _text_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        return int(value) if value else None
    except (OSError, UnicodeError, ValueError):
        return None


def _validate_one(spec: dict[str, Any], input_dir: Path, workspace_root: Path) -> dict[str, Any]:
    name = str(spec["name"])
    directory = input_dir / name
    run_path = directory / "run.jsonl"
    summary_path = directory / "summary.json"
    rows, invalid_json, blank_lines = _read_jsonl(run_path)
    row_issues: list[dict[str, Any]] = []
    # Re-read with line numbers only for issue attribution; this does not alter
    # the source file and keeps invalid line locations explicit.
    try:
        raw_lines = run_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raw_lines = []
    valid_index = 0
    for line_no, line in enumerate(raw_lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(value, dict):
            valid_index += 1
            row_issues.extend(_row_issues(value, line_no))

    summary, summary_error = _read_json(summary_path)
    summary = summary if isinstance(summary, dict) else {}
    attack_rows = [row for row in rows if (row.get("details") or {}).get("is_control") is not True]
    control_rows = [row for row in rows if (row.get("details") or {}).get("is_control") is True]
    attack_successes = sum(row.get("success") is True for row in attack_rows)
    control_passes = sum((row.get("details") or {}).get("control_pass") is True for row in control_rows)
    overblocked = sum((row.get("details") or {}).get("overblocked") is True or row.get("overblocked") is True for row in control_rows)
    attack_asr = attack_successes / len(attack_rows) if attack_rows else None
    control_pass_rate = control_passes / len(control_rows) if control_rows else None
    expected_rows = int(spec["attack_cases"]) + int(spec["control_cases"])
    semantic_issues: list[dict[str, Any]] = []
    if len(rows) != expected_rows:
        semantic_issues.append({"error": "row_count", "expected": expected_rows, "observed": len(rows)})
    if len(attack_rows) != int(spec["attack_cases"]):
        semantic_issues.append({"error": "attack_row_count", "expected": spec["attack_cases"], "observed": len(attack_rows)})
    if len(control_rows) != int(spec["control_cases"]):
        semantic_issues.append({"error": "control_row_count", "expected": spec["control_cases"], "observed": len(control_rows)})
    wrong_ids = sorted({str(row.get("attack_id")) for row in rows if row.get("attack_id") != spec["attack_id"]})
    if wrong_ids:
        semantic_issues.append({"error": "attack_id_mismatch", "values": wrong_ids})
    if summary_error:
        semantic_issues.append({"error": "summary_json", "detail": summary_error})
    else:
        # The official runner's aggregate ``asr`` includes controls (whose
        # success field is false by design).  For executable A2A splits the
        # attack-only value is exposed separately as ``a2a_attack_asr``.
        expected_summary_asr = summary.get("a2a_attack_asr", summary.get("asr"))
        if isinstance(expected_summary_asr, (int, float)) and attack_asr is not None and abs(float(expected_summary_asr) - attack_asr) > 1e-12:
            semantic_issues.append({"error": "summary_asr_mismatch", "summary": expected_summary_asr, "computed": attack_asr})
        if control_rows:
            expected_control = summary.get("control_pass_rate")
            if isinstance(expected_control, (int, float)) and abs(float(expected_control) - control_pass_rate) > 1e-12:
                semantic_issues.append({"error": "summary_control_pass_rate_mismatch", "summary": expected_control, "computed": control_pass_rate})
    execution = _canonical_execution(name, directory, workspace_root)
    execution_ok = execution.get("exit_code") == 0
    if name == "co_eval":
        execution_ok = execution_ok and execution.get("manifest_valid") is True and execution.get("processed_cases") == execution.get("expected_cases") == expected_rows
    required_files = ["run.jsonl", "summary.json"]
    if name == "co_eval":
        required_files.extend(["isolated_command.txt", "isolated_stdout.log", "isolated_stderr.log", "isolated_runner_exit_code.txt", "isolated_manifest.json"])
    else:
        required_files.extend(["command.txt", "stdout.txt", "stderr.txt", "exit_code.txt"])
    missing_files = [_relative(directory / filename, workspace_root) for filename in required_files if not (directory / filename).is_file()]
    valid = not invalid_json and not row_issues and not semantic_issues and execution_ok and not missing_files
    return {
        "name": name,
        "attack_id": spec["attack_id"],
        "config": spec["config"],
        "canonical": valid,
        "valid": valid,
        "run_path": _relative(run_path, workspace_root),
        "summary_path": _relative(summary_path, workspace_root),
        "run_sha256": _sha256(run_path),
        "line_count": len(raw_lines),
        "blank_lines": blank_lines,
        "valid_row_count": len(rows),
        "invalid_json_lines": invalid_json,
        "row_schema_issues": row_issues,
        "semantic_issues": semantic_issues,
        "missing_files": missing_files,
        "attack_rows": len(attack_rows),
        "attack_successes": attack_successes,
        "attack_asr": attack_asr,
        "control_rows": len(control_rows),
        "control_passes": control_passes,
        "control_pass_rate": control_pass_rate,
        "overblocked_controls": overblocked,
        "overblock_rate": overblocked / len(control_rows) if control_rows else None,
        "event_count": sum(len(((row.get("details") or {}).get("events") or [])) for row in rows if isinstance((row.get("details") or {}).get("events", []), list)),
        "execution": execution,
        "summary_snapshot": {
            key: summary.get(key)
            for key in ("asr", "a2a_attack_asr", "a2a_attack_trials", "control_pass_rate", "control_trials", "overblock_rate", "total_trials", "total_successes", "config_hash")
            if key in summary
        },
    }


def _git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _excluded_legacy(input_dir: Path, workspace_root: Path) -> list[dict[str, Any]]:
    excluded: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        rows, invalid, blanks = _read_jsonl(path)
        excluded.append(
            {
                "path": _relative(path, workspace_root),
                "reason": "legacy top-level snapshot; canonical subdirectory run is authoritative",
                "line_count": len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0,
                "valid_rows": len(rows),
                "invalid_json_lines": invalid,
                "blank_lines": blanks,
                "sha256": _sha256(path),
            }
        )
    return excluded


def _markdown(manifest: dict[str, Any]) -> str:
    rows = manifest["runs"]
    lines = [
        "# A2ASecBench v1 offline baseline validation",
        "",
        '本报告只使用 `paperAlpha/experiments/a2asecbench/runs/a2asecbench_v1/{as,asrf_eval,atsi_eval,co_eval,hotf_eval}/run.jsonl` 作为 canonical 输入。每个 JSONL 已逐行解析，并将非法行位置保存在机器可读 manifest 中。',
        "",
        "这些数值是 A2ASecBench 的离线 reference SUT baseline 快照，不是我们的模型、监测器或 BN 结果。攻击成功率（ASR）与控制通过率分开计算：攻击行看 `success`，控制行看显式 `details.control_pass`。",
        "",
        "| 配置 | canonical 行数 | 事件数 | 非法 JSON 行 | 攻击 ASR | 控制通过率 | canonical 状态 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in rows:
        asr = "—" if run["attack_asr"] is None else f"{run['attack_asr']:.3f}"
        control = "—" if run["control_pass_rate"] is None else f"{run['control_pass_rate']:.3f}"
        status = "通过" if run["valid"] else "失败"
        lines.append(f"| {run['attack_id']} (`{run['name']}`) | {run['valid_row_count']} | {run['event_count']} | {len(run['invalid_json_lines'])} | {asr} | {control} | {status} |")
    lines += [
        "",
        "## 运行证据",
        "",
    ]
    for run in rows:
        execution = run["execution"]
        # The report lives inside the results directory, so links should be
        # relative to that directory.  The manifest retains workspace-relative
        # paths for unambiguous machine consumption.
        run_link = f"{run['name']}/run.jsonl"
        summary_link = f"{run['name']}/summary.json"
        lines += [
            f"### {run['attack_id']} (`{run['name']}`)",
            "",
            f"- Config: `{run['config']}`；trials override: `1`。",
            f"- Run: [`{run_link}`]({run_link})",
            f"- Summary: [`{summary_link}`]({summary_link})",
            f"- SHA-256: `{run['run_sha256']}`",
            f"- Command: `{execution.get('command')}`；stdout: `{execution.get('stdout')}`；exit: `{execution.get('exit_code')}`。",
        ]
        if run["name"] == "co_eval":
            original = execution.get("original_attempt", {})
            lines += [
                f"- CO 使用逐 case 隔离恢复：manifest `{execution.get('manifest')}`，{execution.get('processed_cases')}/{execution.get('expected_cases')} cases 有效，单 case timeout `{execution.get('timeout_s')}s`。",
                f"- 初次 aggregate 尝试的状态文件为 exit `{original.get('exit_code')}`，但该尝试在控制样本 032 前停止且 stdout/stderr 为空；其结果明确排除，不参与上表。",
            ]
        lines += [
            f"- 攻击行 `{run['attack_successes']}/{run['attack_rows']}`；控制行 `{run['control_passes']}/{run['control_rows']}`；观测到的 control overblock 行 `{run['overblocked_controls']}`。",
            "",
        ]
    lines += [
        "## Layer 2 动态图映射",
        "",
        '适配器 [`paperAlpha/code/commands/adapt_a2asecbench.py`](../../scripts/adapt_a2asecbench.py) 只读取上述 canonical run。每个 benchmark row 保留为一个 episode；`details.events[]` 每项投影为一个 Layer 2 event。',
        "",
        "- 节点候选：显式 `actor_id`、`evidence.agent_id`/`next_agent_id`、`component`；不从工具名推断额外节点。",
        "- 边候选：显式 `parent_task_id`、delegation 的 source/target、A2A message 的 source/target；仅有时间顺序时保留 `sequence`，不声称因果边。",
        "- 证据：`public_surface`、`evidence`、`private_tags`（脱敏后）、metrics/benchmark outcome flags。`timestamp_monotonic` 原样保留为相对/单调时间。",
        "- `scenario_role`（attack/control）与 `observed_violation` 分离；没有显式 per-event risk label 时 `risk_label` 保持 null，不把 benchmark attack 身份冒充标签。",
        "",
        "## 边界与排除项",
        "",
        "- 顶层 `AS.jsonl`、`ASRF.jsonl`、`ATSI.jsonl`、`CO.jsonl`、`HOTF.jsonl` 是旧快照，未用于本报告；它们的行完整性另列在 `A2ASECBENCH_MANIFEST.json`。",
        "- 离线 reference SUT 的高 ASR 只能说明该 benchmark/SUT 组合的基线行为，不能推出我们的模型效果，也不能作为因果结构真值。",
        "- A2A 事件可供第二层动态执行图回放；正式风险评估仍需带显式标签且经过版本锁定的数据划分。",
        "",
        "机器可读证据：[`A2ASECBENCH_MANIFEST.json`](A2ASECBENCH_MANIFEST.json)。",
        "",
    ]
    return "\n".join(lines)


def validate(input_dir: Path = DEFAULT_INPUT, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or input_dir
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    workspace_root = ROOT.parent.resolve()
    runs = [_validate_one(spec, input_dir, workspace_root) for spec in RUN_SPECS]
    source_repo = ROOT / 'code/external/A2ASecBench'
    manifest = {
        "schema_version": "a2asecbench-v1-validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": _relative(input_dir, workspace_root),
        "source_repo": _relative(source_repo, workspace_root),
        "source_commit": _git_commit(source_repo),
        "canonical_policy": "Only fixed subdirectory run.jsonl files are authoritative; legacy top-level snapshots are excluded.",
        "metric_policy": {
            "attack_asr": "mean(success == true) over rows with details.is_control != true",
            "control_pass_rate": "mean(details.control_pass == true) over rows with details.is_control == true",
            "overblock_rate": "mean(overblocked == true) over control rows; row-level or details-level field",
        },
        "runs": runs,
        "excluded_legacy_top_level": _excluded_legacy(input_dir, workspace_root),
        "all_canonical_valid": all(run["valid"] for run in runs),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "A2ASECBENCH_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "A2ASECBENCH_REPORT.md").write_text(_markdown(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest = validate(args.input, args.output)
    print(json.dumps({"all_canonical_valid": manifest["all_canonical_valid"], "runs": [{"name": r["name"], "valid": r["valid"], "rows": r["valid_row_count"], "invalid": len(r["invalid_json_lines"])} for r in manifest["runs"]]}, ensure_ascii=False))
    raise SystemExit(0 if manifest["all_canonical_valid"] else 1)


if __name__ == "__main__":
    main()
