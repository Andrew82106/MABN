"""Future live-only Q2-C runner, implemented and tested offline with fake HTTP.

Nothing in this module performs I/O until an explicit caller invokes the live
execution function.  The CLI is the only path that reads the local credential
file, and it must first receive the exact release flag.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import time
from typing import Any, Callable

from .config import (
    BASE_URL_IDENTITY,
    TARGET_MODEL,
    TOTAL_DEADLINE_SECONDS,
    canonical_json,
    profile_commitment,
    public_profile_snapshot,
    schema_version_for_run_kind,
    sha256_json,
)
from .evidence import (
    artifact_manifest,
    assert_artifact_text_safe,
    build_ledger,
    derive_outcome,
    derive_report,
    local_delivery_text,
    run_directory,
    write_json,
    write_text,
)
from .historical_integrity import historical_tree_snapshot
from .live_transport import (
    ConnectionFactory,
    LiveConfiguration,
    LiveHTTPTransport,
    LIVE_CHAT_COMPLETIONS_PATH,
    LIVE_MODELS_PATH,
    default_connection_factory,
    validate_live_configuration,
)
from .runner import execute_protocol


LIVE_GUARD_NAME = "live_probe_once_guard.json"
LIVE_VARIABLE_NAMES = (
    "P1V2_Q2_BASE_URL",
    "P1V2_Q2_API_KEY",
    "P1V2_Q2_MODEL",
)


def _module_sha256(module_name: str) -> str:
    source = Path(__file__).with_name(module_name).read_bytes()
    return sha256(source).hexdigest()


def live_execution_source_hashes() -> dict[str, str]:
    """Hash the complete local source closure used by the released live CLI."""
    package_root = Path(__file__).resolve().parent
    code_root = package_root.parent
    paths = [
        *sorted(package_root.glob("*.py"), key=lambda item: item.name),
        code_root / "scripts" / "run_q2c_probe.py",
    ]
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("live_execution_source_unavailable")
        relative = path.relative_to(code_root).as_posix()
        hashes[relative] = sha256(path.read_bytes()).hexdigest()
    return hashes


def live_runner_source_sha256() -> str:
    return live_execution_source_hashes()["q2c_probe/live_runner.py"]


def live_transport_source_sha256() -> str:
    return live_execution_source_hashes()["q2c_probe/live_transport.py"]


def live_source_bundle_sha256() -> str:
    return sha256_json(live_execution_source_hashes())


def live_policy_snapshot() -> dict[str, Any]:
    return {
        "base_url_identity": BASE_URL_IDENTITY,
        "models_endpoint": LIVE_MODELS_PATH,
        "chat_completions_endpoint": LIVE_CHAT_COMPLETIONS_PATH,
        "target_model": TARGET_MODEL,
        "metadata_call_max": 1,
        "completion_call_max": 4,
        "concurrency": 1,
        "retry_count": 0,
        "stream": False,
        "total_deadline_seconds": TOTAL_DEADLINE_SECONDS,
    }


def live_policy_commitment() -> str:
    return sha256_json(live_policy_snapshot())


def live_config_snapshot() -> dict[str, Any]:
    """Artifact-safe frozen configuration for a live probe (never credentials)."""
    return {
        "schema_version": schema_version_for_run_kind("live_probe"),
        "run_kind": "live_probe",
        "offline_only": False,
        "live_only": True,
        "profile_commitment": profile_commitment("live_probe"),
        "public_profile": public_profile_snapshot("live_probe"),
        "live_policy_commitment": live_policy_commitment(),
        "live_runner_source_sha256": live_runner_source_sha256(),
        "live_transport_source_sha256": live_transport_source_sha256(),
        "live_source_bundle_sha256": live_source_bundle_sha256(),
        "live_execution_source_hashes": live_execution_source_hashes(),
        "live_policy": live_policy_snapshot(),
        "credential_variable_count": 3,
    }


def make_live_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"P1V2Q2C-PROBE-{timestamp}"


def _guard_path(data_root: Path) -> Path:
    return data_root / LIVE_GUARD_NAME


def _existing_live_attempt(data_root: Path) -> bool:
    guard = _guard_path(data_root)
    if guard.exists() or guard.is_symlink():
        return True
    runs_root = data_root / "runs"
    if not runs_root.exists():
        return False
    if runs_root.is_symlink() or not runs_root.is_dir():
        return True
    for candidate in runs_root.iterdir():
        if candidate.is_symlink() or not candidate.is_dir():
            return True
        if candidate.name.startswith("P1V2Q2C-PROBE-"):
            return True
        manifest_path = candidate / "run_manifest.json"
        if manifest_path.is_file() and not manifest_path.is_symlink():
            try:
                import json

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return True
            if isinstance(manifest, dict) and manifest.get("run_kind") == "live_probe":
                return True
    return False


def reserve_live_attempt(data_root: Path, run_id: str) -> Path:
    """Atomically consume the one live attempt before credentials or I/O are used."""
    run_directory(data_root, run_id)
    if _existing_live_attempt(data_root):
        raise RuntimeError("live_probe_already_consumed")
    data_root.mkdir(parents=True, exist_ok=True)
    guard = _guard_path(data_root)
    payload = {
        "schema_version": schema_version_for_run_kind("live_probe"),
        "run_id": run_id,
        "state": "reserved",
        "live_policy_commitment": live_policy_commitment(),
    }
    serialized = canonical_json(payload) + "\n"
    assert_artifact_text_safe(serialized)
    try:
        with guard.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    except FileExistsError:
        raise RuntimeError("live_probe_already_consumed") from None
    return guard


def _read_exact_credential_values(dotenv_path: Path) -> LiveConfiguration:
    """Live-path-only parser that selects precisely the three allowed variable names."""
    values: dict[str, str] = {}
    try:
        with dotenv_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name = name.strip()
                if name not in LIVE_VARIABLE_NAMES:
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                    value = value[1:-1]
                values[name] = value
    except OSError:
        raise RuntimeError("live_credential_unavailable") from None
    if set(values) != set(LIVE_VARIABLE_NAMES):
        raise RuntimeError("live_credential_missing")
    configuration = LiveConfiguration(
        base_url=values["P1V2_Q2_BASE_URL"],
        api_key=values["P1V2_Q2_API_KEY"],
        model=values["P1V2_Q2_MODEL"],
    )
    values.clear()
    validate_live_configuration(configuration)
    return configuration


def load_live_configuration_from_dotenv(dotenv_path: Path) -> LiveConfiguration:
    """Explicit wrapper used solely by the released CLI live branch."""
    return _read_exact_credential_values(dotenv_path)


def _persist_live_transcript(
    data_root: Path,
    transcript: dict[str, Any],
    history_before: dict[str, Any],
    history_after: dict[str, Any],
    root_guard: Path,
) -> Path:
    run_id = transcript["run_id"]
    run_dir = run_directory(data_root, run_id)
    if run_dir.exists():
        raise FileExistsError("live_run_directory_exists")
    if history_before != history_after:
        raise RuntimeError("historical_tree_changed")
    try:
        guard_payload = root_guard.read_text(encoding="utf-8")
    except OSError:
        raise RuntimeError("live_guard_missing") from None
    assert_artifact_text_safe(guard_payload)
    run_dir.mkdir(parents=True, exist_ok=False)
    config_snapshot = live_config_snapshot()
    history = {"before": history_before, "after": history_after, "unchanged": True}
    ledger = build_ledger(transcript)
    outcome = derive_outcome(transcript)
    report = derive_report(transcript, outcome)
    write_json(run_dir / "config_snapshot.json", config_snapshot)
    write_json(run_dir / "transcript.json", transcript)
    write_json(run_dir / "ledger.json", ledger)
    write_json(run_dir / "outcome.json", outcome)
    write_json(run_dir / "report.json", report)
    write_json(run_dir / "historical_integrity.json", history)
    write_text(run_dir / LIVE_GUARD_NAME, guard_payload)
    write_text(run_dir / "reports" / "DELIVERY.md", local_delivery_text(run_id, outcome))
    manifest_extra = {
        "live_policy_commitment": live_policy_commitment(),
        "live_runner_source_sha256": live_runner_source_sha256(),
        "live_transport_source_sha256": live_transport_source_sha256(),
        "live_source_bundle_sha256": live_source_bundle_sha256(),
        "live_execution_source_hashes": live_execution_source_hashes(),
        "metadata_call_count": transcript["metadata_call_count"],
        "completion_call_count": transcript["completion_call_count"],
    }
    write_json(run_dir / "run_manifest.json", artifact_manifest(run_dir, run_id, "live_probe", manifest_extra))
    return run_dir


def execute_live_probe_with_configuration(
    data_root: Path,
    configuration: LiveConfiguration,
    *,
    connection_factory: ConnectionFactory = default_connection_factory,
    run_id: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[str, Path]:
    """Run the frozen live protocol. Tests inject fake HTTP; production needs CLI release."""
    validate_live_configuration(configuration)
    actual_run_id = run_id or make_live_run_id()
    root_guard = reserve_live_attempt(data_root, actual_run_id)
    before = historical_tree_snapshot()
    transport = LiveHTTPTransport(configuration, connection_factory)
    try:
        transcript = execute_protocol(
            transport,
            run_id=actual_run_id,
            run_kind="live_probe",
            total_deadline_seconds=TOTAL_DEADLINE_SECONDS,
            clock=clock,
        )
    finally:
        transport.clear_secret()
    after = historical_tree_snapshot()
    transcript["live_policy_commitment"] = live_policy_commitment()
    transcript["live_runner_source_sha256"] = live_runner_source_sha256()
    transcript["live_transport_source_sha256"] = live_transport_source_sha256()
    transcript["metadata_call_count"] = sum(
        1 for call in transport.calls if call.get("kind") == "metadata"
    )
    transcript["completion_call_count"] = sum(
        1 for call in transport.calls if call.get("kind") == "completion"
    )
    if transcript["metadata_call_count"] > 1 or transcript["completion_call_count"] > 4:
        raise RuntimeError("live_call_budget_violation")
    run_dir = _persist_live_transcript(data_root, transcript, before, after, root_guard)
    return actual_run_id, run_dir


def execute_released_live_probe(data_root: Path, dotenv_path: Path) -> tuple[str, Path]:
    """The CLI-only live branch: reserve first, then briefly read exact credentials."""
    run_id = make_live_run_id()
    reserve_live_attempt(data_root, run_id)
    configuration = load_live_configuration_from_dotenv(dotenv_path)
    # A second reservation is intentionally avoided; execute using the already consumed guard.
    return _execute_reserved_live_probe(data_root, configuration, run_id)


def _execute_reserved_live_probe(
    data_root: Path,
    configuration: LiveConfiguration,
    run_id: str,
    *,
    connection_factory: ConnectionFactory = default_connection_factory,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[str, Path]:
    validate_live_configuration(configuration)
    root_guard = _guard_path(data_root)
    before = historical_tree_snapshot()
    transport = LiveHTTPTransport(configuration, connection_factory)
    try:
        transcript = execute_protocol(
            transport,
            run_id=run_id,
            run_kind="live_probe",
            total_deadline_seconds=TOTAL_DEADLINE_SECONDS,
            clock=clock,
        )
    finally:
        transport.clear_secret()
    after = historical_tree_snapshot()
    transcript["live_policy_commitment"] = live_policy_commitment()
    transcript["live_runner_source_sha256"] = live_runner_source_sha256()
    transcript["live_transport_source_sha256"] = live_transport_source_sha256()
    transcript["metadata_call_count"] = sum(1 for call in transport.calls if call.get("kind") == "metadata")
    transcript["completion_call_count"] = sum(1 for call in transport.calls if call.get("kind") == "completion")
    if transcript["metadata_call_count"] > 1 or transcript["completion_call_count"] > 4:
        raise RuntimeError("live_call_budget_violation")
    run_dir = _persist_live_transcript(data_root, transcript, before, after, root_guard)
    return run_id, run_dir
