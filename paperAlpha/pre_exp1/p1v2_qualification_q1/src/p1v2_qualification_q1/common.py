"""Paths, strict serialization, provenance, and append-only artifact helpers.

This module deliberately has no dependency on Q0 Python modules.  The only Q0
reads are the seven literal source-contract files enumerated in the frozen
``source_contracts.json`` file.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .errors import PreflightError, ValidationError


PACKAGE_FILE = Path(__file__).resolve()
PROJECT_ROOT = PACKAGE_FILE.parents[5]
CODE_ROOT = PROJECT_ROOT / "paperAlpha" / "pre_exp1" / "p1v2_qualification_q1"
DATA_ROOT = PROJECT_ROOT / "paperAlpha" / "data" / "pre_exp1" / "p1v2_qualification_q1"
RUNS_ROOT = DATA_ROOT / "runs"
RUN_ID_RE = re.compile(r"^P1V2Q1-LOCAL-\d{8}T\d{12}Z$")

IDENTITY_FALSE_FIELDS = (
    "is_new_p1_run",
    "eligible_for_p1_gate_analysis",
    "eligible_for_p2",
    "eligible_for_confirmatory_analysis",
    "eligible_for_causal_effect_analysis",
    "p1_go",
    "p2_allowed",
)

EXPECTED_SOURCE_CONTRACTS = (
    ("paperAlpha/pre_exp1/p1v2_qualification/configs/q1_protocol.json", "provenance/q0_q1_protocol.json", "2cbc4eca4e0cbef597d548510424d3441bab9eb17438ef208dc488237346c87e"),
    ("paperAlpha/pre_exp1/p1v2_qualification/schemas/response_contract.schema.json", "schemas/response_contract.schema.json", "ec2534040ba43eab83878a48aeb953314379f3fb6d9ed6896ae20fa4f7afbf94"),
    ("paperAlpha/pre_exp1/p1v2_qualification/prompts/public_response_contract.txt", "prompts/public_response_contract.txt", "ec742f53fb9960350c0ced5195480533a76ab3eaa6929b87aaf106a02486223d"),
    ("paperAlpha/pre_exp1/p1v2_qualification/fixtures/screen_tasks.json", "fixtures/screen_tasks.json", "ee0166d382a709ae96aeb99fce2bbb55f2c2ce5a42ca29ab5fc0d2ecbe4b2084"),
    ("paperAlpha/pre_exp1/p1v2_qualification/fixtures/confirmation_tasks.json", "fixtures/confirmation_tasks.json", "f6cd3ae51f56688e25489985f14aa5ca2434e5e81983338fe0b9d5e08ec7df57"),
    ("paperAlpha/pre_exp1/p1v2_qualification/permissions/public_sink_rule.json", "permissions/public_sink_rule.json", "e09d72755a522db616f1d0a94848c9589242fbdcbae7f0b6e9855d3f158cc02c"),
    ("paperAlpha/data/pre_exp1/p1v2_qualification/readiness_dry/manifests/manifest_P1V2Q-READINESS-DRY-20260801T010106000000Z.json", "provenance/q0_current_manifest.json", "e3e2772c3a96031813464426746717038bcc4f13571d8b8dadb98650bf4d4e4f"),
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _reject_constant(token: str) -> None:
    raise ValueError(f"non_finite_json_constant:{token}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    if not isinstance(text, str):
        raise ValueError("json_text_not_string")
    if not text.strip():
        raise ValueError("empty_json_text")
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_constant,
    )


def load_json(path: Path) -> Any:
    try:
        return strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid_json:{path.name}:{type(exc).__name__}") from exc


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ValidationError("invalid_run_id")
    return run_id


def new_run_id() -> str:
    return "P1V2Q1-LOCAL-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def ensure_relative(base: Path, candidate: Path, label: str) -> Path:
    resolved_base = base.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_base)
    except ValueError as exc:
        raise ValidationError(f"unsafe_{label}_path") from exc
    return resolved_candidate


def safe_relative_path(root: Path, relative_path: str, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValidationError(f"invalid_{label}_path")
    normalized = relative_path.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("//") or ":" in normalized:
        raise ValidationError(f"unsafe_{label}_path")
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValidationError(f"unsafe_{label}_path")
    if any(part.lower() == ".env" for part in parts):
        raise ValidationError(f"forbidden_{label}_path")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{index}" for index in range(1, 10)), *(f"LPT{index}" for index in range(1, 10))}
    if any(part.split(".", 1)[0].upper() in reserved for part in parts):
        raise ValidationError(f"device_{label}_path")
    return ensure_relative(root, root.joinpath(*parts), label)


def run_dir_for(run_id: str) -> Path:
    validate_run_id(run_id)
    return ensure_relative(RUNS_ROOT, RUNS_ROOT / run_id, "run")


def artifact_paths(run_id: str) -> dict[str, Path]:
    run_dir = run_dir_for(run_id)
    return {
        "run_dir": run_dir,
        "events": run_dir / "events" / f"events_{run_id}.jsonl",
        "transcripts": run_dir / "transcripts" / f"transcripts_{run_id}.jsonl",
        "public_sink": run_dir / "public_sink" / f"public_sink_{run_id}.jsonl",
        "metadata": run_dir / "metadata" / f"metadata_{run_id}.jsonl",
        "unstarted": run_dir / "unstarted" / f"unstarted_{run_id}.jsonl",
        "outcome": run_dir / "outcomes" / f"outcomes_{run_id}.json",
        "manifest": run_dir / "manifests" / f"manifest_{run_id}.json",
        "validation": run_dir / "validation" / f"validation_{run_id}.jsonl",
        "replay": run_dir / "replay" / f"replay_{run_id}.jsonl",
        "delivery": run_dir / "reports" / "DELIVERY.md",
    }


def initialize_run(run_id: str) -> dict[str, Path]:
    paths = artifact_paths(run_id)
    run_dir = paths["run_dir"]
    if run_dir.exists():
        raise PreflightError("run_id_collision")
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(mode=0o700)
    for key, path in paths.items():
        if key == "run_dir":
            continue
        path.parent.mkdir(parents=True, exist_ok=False)
    for key in ("events", "transcripts", "public_sink", "metadata", "unstarted", "validation", "replay"):
        paths[key].touch(exist_ok=False)
    return paths


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_relative(DATA_ROOT, path, "artifact")
    encoded = canonical_json_bytes(record).decode("utf-8") + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    ensure_relative(DATA_ROOT, path, "artifact")
    encoded = canonical_json_bytes(value).decode("utf-8") + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise ValidationError(f"append_only_overwrite_refused:{path.name}") from exc


def write_text_once(path: Path, text: str) -> None:
    ensure_relative(DATA_ROOT, path, "artifact")
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise ValidationError(f"append_only_overwrite_refused:{path.name}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            record = strict_json_loads(line)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid_jsonl:{path.name}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValidationError(f"non_object_jsonl:{path.name}:{line_number}")
        records.append(record)
    return records


def file_map(root: Path, include: Iterable[Path] | None = None) -> dict[str, str]:
    if include is None:
        candidates = [path for path in root.rglob("*") if path.is_file()]
    else:
        candidates = list(include)
    result: dict[str, str] = {}
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        if relative.lower().endswith(".pyc"):
            continue
        result[relative] = sha256_file(path)
    return result


def mapping_hash(mapping: dict[str, str]) -> str:
    return sha256_bytes(canonical_json_bytes(mapping))


def code_provenance() -> dict[str, Any]:
    source_files = file_map(CODE_ROOT)
    return {"source_files": source_files, "source_tree_hash": mapping_hash(source_files)}


def static_input_hashes() -> dict[str, str]:
    input_roots = (
        CODE_ROOT / "configs",
        CODE_ROOT / "schemas",
        CODE_ROOT / "prompts",
        CODE_ROOT / "fixtures",
        CODE_ROOT / "permissions",
        CODE_ROOT / "provenance",
    )
    files: list[Path] = []
    for root in input_roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return file_map(CODE_ROOT, files)


def load_protocol() -> dict[str, Any]:
    protocol = load_json(CODE_ROOT / "configs" / "q1_live_protocol.json")
    if not isinstance(protocol, dict):
        raise PreflightError("protocol_not_object")
    required = {
        "artifact_kind": "p1v2_model_output_qualification",
        "phase": "P1V2_Q1_LIVE_LOCAL",
        "data_role": "p1v2_qualification_live",
        "execution_mode": "live_local_qualification",
        "model_output_stack_id": "QWEN3_STRICT_JSON_V1",
        "model_tag": "qwen3:8b",
        "provider": "local_ollama_loopback",
        "endpoint": "http://127.0.0.1:11434/api/chat",
        "format": "json",
        "temperature": 0,
        "seed": 20260801,
        "think": False,
        "num_ctx": 2048,
        "max_output_tokens": 128,
        "concurrency": 1,
        "retry_per_call": 0,
        "per_call_timeout_seconds": 60,
        "screen_wall_clock_limit_minutes": 20,
        "confirmation_wall_clock_limit_minutes": 45,
        "fixed_q0_run_id": "P1V2Q-READINESS-DRY-20260801T010106000000Z",
        "q0_source_contract_sha256": "8a65bafd28e704c6f4aa7d719e522b93e8daf6a9d0d603f30f55dde0e8942ed3",
    }
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise PreflightError(f"protocol_mismatch:{key}")
    if protocol.get("screen_counts") != {"coordinator": 16, "publisher": 16}:
        raise PreflightError("protocol_mismatch:screen_counts")
    if protocol.get("confirmation_counts") != {"coordinator": 32, "publisher": 64}:
        raise PreflightError("protocol_mismatch:confirmation_counts")
    identity = protocol.get("identity")
    if not isinstance(identity, dict):
        raise PreflightError("protocol_identity_missing")
    for key in IDENTITY_FALSE_FIELDS:
        if identity.get(key) is not False:
            raise PreflightError(f"protocol_identity_not_false:{key}")
    return protocol


def _source_contracts() -> dict[str, Any]:
    contracts = load_json(CODE_ROOT / "provenance" / "source_contracts.json")
    if not isinstance(contracts, dict) or not isinstance(contracts.get("allowed_q0_sources"), list):
        raise ValidationError("invalid_source_contracts")
    if contracts.get("q0_run_id") != "P1V2Q-READINESS-DRY-20260801T010106000000Z":
        raise ValidationError("source_contract_q0_run_id")
    if len(contracts["allowed_q0_sources"]) != 7:
        raise ValidationError("source_contract_count")
    observed = tuple(
        (item.get("q0_relative_path"), item.get("frozen_relative_path"), item.get("sha256"))
        for item in contracts["allowed_q0_sources"]
        if isinstance(item, dict)
    )
    if observed != EXPECTED_SOURCE_CONTRACTS:
        raise ValidationError("source_contract_identity")
    return contracts


def verify_source_contracts() -> list[dict[str, str]]:
    contracts = _source_contracts()
    observed: list[dict[str, str]] = []
    for item in contracts["allowed_q0_sources"]:
        if not isinstance(item, dict):
            raise ValidationError("invalid_source_contract_entry")
        q0_relative = item.get("q0_relative_path")
        frozen_relative = item.get("frozen_relative_path")
        expected_hash = item.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValidationError("invalid_source_contract_hash")
        q0_path = safe_relative_path(PROJECT_ROOT, q0_relative, "q0_source")
        frozen_path = safe_relative_path(CODE_ROOT, frozen_relative, "frozen_source")
        if not q0_path.is_file() or not frozen_path.is_file():
            raise ValidationError("missing_source_contract_file")
        q0_hash = sha256_file(q0_path)
        frozen_hash = sha256_file(frozen_path)
        if q0_hash != expected_hash or frozen_hash != expected_hash:
            raise ValidationError("source_contract_drift")
        observed.append(
            {
                "q0_relative_path": q0_relative,
                "frozen_relative_path": frozen_relative,
                "sha256": expected_hash,
            }
        )
    return observed


def q0_manifest_core() -> dict[str, Any]:
    manifest = load_json(CODE_ROOT / "provenance" / "q0_current_manifest.json")
    if not isinstance(manifest, dict):
        raise ValidationError("q0_manifest_not_object")
    if manifest.get("run_id") != "P1V2Q-READINESS-DRY-20260801T010106000000Z":
        raise ValidationError("q0_manifest_run_id")
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ValidationError("q0_manifest_artifact_hashes")
    selected = {key: artifact_hashes.get(key) for key in ("events", "transcripts", "outcomes", "public_sink")}
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in selected.values()):
        raise ValidationError("q0_manifest_core_hashes")
    return {"run_id": manifest["run_id"], "artifact_hashes": selected, "manifest_sha256": sha256_file(CODE_ROOT / "provenance" / "q0_current_manifest.json")}


def fixed_identity() -> dict[str, bool]:
    return {key: False for key in IDENTITY_FALSE_FIELDS}
