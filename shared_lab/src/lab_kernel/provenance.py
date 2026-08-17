"""Content-addressed run provenance and trusted receipt verification."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import content_hash, file_hash, tree_hash
from .ledger import verify_ledger
from .state import state_hash


class ProvenanceError(RuntimeError):
    pass


def environment_record() -> dict[str, Any]:
    packages = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    record = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }
    return {**record, "record_hash": content_hash(record)}


def _combined_code_hash(roots: Iterable[Path]) -> tuple[str, dict[str, Any]]:
    detail: dict[str, Any] = {}
    content_identity: list[dict[str, Any]] = []
    for index, root in enumerate(roots):
        digest, hashes = tree_hash(root, (p for p in root.rglob("*.py") if p.is_file()))
        detail[str(index)] = {
            "recorded_path": str(root.resolve()), "tree_hash": digest, "files": hashes
        }
        content_identity.append({"tree_hash": digest, "files": hashes})
    return content_hash(content_identity), detail


def build_provenance(scenario_root: Path, code_roots: Iterable[Path]) -> dict[str, Any]:
    scenario_digest, scenario_files = tree_hash(scenario_root)
    code_digest, code_detail = _combined_code_hash(code_roots)
    environment = environment_record()
    body = {
        "schema_version": 1,
        "scenario": {
            "root": str(scenario_root.resolve()),
            "tree_hash": scenario_digest,
            "files": scenario_files,
        },
        "code": {"tree_hash": code_digest, "roots": code_detail},
        "environment": environment,
    }
    return {**body, "provenance_hash": content_hash(body)}


def write_receipt(
    path: Path,
    provenance: Mapping[str, Any],
    *,
    run_id: str,
    ledger_path: Path,
    ledger_head: str,
    event_count: int,
    final_state: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "run_id": run_id,
        "provenance": dict(provenance),
        "ledger": {
            "path": str(ledger_path.resolve()),
            "file_hash": file_hash(ledger_path),
            "head_hash": ledger_head,
            "event_count": event_count,
        },
        "final_state_hash": state_hash(dict(final_state)),
    }
    receipt = {**body, "receipt_hash": content_hash(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path.resolve(), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return receipt


def load_receipt(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    claimed = receipt.pop("receipt_hash", None)
    actual = content_hash(receipt)
    receipt["receipt_hash"] = claimed
    if claimed != actual:
        raise ProvenanceError("Receipt hash mismatch")
    return receipt


def verify_receipt(
    receipt: Mapping[str, Any],
    *,
    scenario_root: Path,
    code_roots: Iterable[Path],
    ledger_path: Path,
    check_current_environment: bool = False,
) -> dict[str, Any]:
    copy_without_hash = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if receipt.get("receipt_hash") != content_hash(copy_without_hash):
        raise ProvenanceError("Receipt hash mismatch")
    recorded = receipt["provenance"]
    recorded_body = {key: value for key, value in recorded.items() if key != "provenance_hash"}
    if recorded.get("provenance_hash") != content_hash(recorded_body):
        raise ProvenanceError("Provenance hash mismatch")
    current = build_provenance(scenario_root, code_roots)
    if current["scenario"]["tree_hash"] != recorded["scenario"]["tree_hash"]:
        raise ProvenanceError("Scenario/configuration content changed")
    if current["code"]["tree_hash"] != recorded["code"]["tree_hash"]:
        raise ProvenanceError("Kernel or hook code changed")
    environment = recorded["environment"]
    env_body = {key: value for key, value in environment.items() if key != "record_hash"}
    if environment.get("record_hash") != content_hash(env_body):
        raise ProvenanceError("Recorded environment was tampered")
    if check_current_environment and current["environment"]["record_hash"] != environment["record_hash"]:
        raise ProvenanceError("Current environment differs from recorded environment")
    ledger = receipt["ledger"]
    result = verify_ledger(
        ledger_path, expected_run_id=receipt["run_id"], expected_head=ledger["head_hash"],
        expected_file_hash=ledger["file_hash"],
    )
    if result["event_count"] != ledger["event_count"]:
        raise ProvenanceError("Ledger event count mismatch")
    return {"valid": True, "run_id": receipt["run_id"], "provenance_hash": recorded["provenance_hash"]}
