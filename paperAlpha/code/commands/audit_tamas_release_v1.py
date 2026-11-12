"""Audit and boundary-safe projection for the official TAMAS release.

This script does *not* run an agent, infer attack labels from text, or train a
monitor.  It verifies the static release layout and writes two explicitly
separated files:

* ``cases_public.jsonl``: the fields a downstream runner may expose to the
  model/monitor (domain, role names, and the user query).  It contains no
  attack category or binary label.
* ``labels.jsonl``: evaluator-only metadata (attack category and source row).

The official TAMAS release contains prompts and role descriptions, not runtime
message/tool traces.  Therefore this projection alone is not a valid input to
the API-only monitor; the official AutoGen/CrewAI runner must first produce
runtime logs, which can then be adapted under the existing event schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ATTACK_DIRS = {"Byzantine", "Colluding", "Contradicting", "DPI", "Impersonation", "IPI"}
REQUIRED_CASE_KEYS = {"agents", "user query"}


def _json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return value


def _domain_from_file(path: Path) -> str:
    # The release uses ``news_contradinting.json`` (sic) for one file, so
    # derive the domain from the prefix rather than from the attack spelling.
    stem = path.stem
    return stem.split("_", 1)[0]


def _case_id(source: str, row_index: int) -> str:
    digest = hashlib.sha256(f"{source}#{row_index}".encode("utf-8")).hexdigest()[:16]
    return f"tamas-{digest}"


def _iter_cases(data_root: Path) -> Iterable[tuple[str, str, Path, int, dict[str, Any]]]:
    for attack_dir in sorted(data_root.iterdir()):
        if not attack_dir.is_dir() or attack_dir.name not in ATTACK_DIRS:
            continue
        for path in sorted(attack_dir.glob("*.json")):
            attack = attack_dir.name
            domain = _domain_from_file(path)
            for row_index, row in enumerate(_json_rows(path)):
                yield attack, domain, path, row_index, row


def audit(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    data_root = repo_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"missing official TAMAS data directory: {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    public_path = output_dir / "cases_public.jsonl"
    labels_path = output_dir / "labels.jsonl"
    manifest_path = output_dir / "manifest.json"
    rows = list(_iter_cases(data_root))
    if not rows:
        raise ValueError("no TAMAS attack files found")

    attack_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    agent_count: Counter[int] = Counter()
    field_shapes: Counter[str] = Counter()
    public_records: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []

    for attack, domain, path, row_index, row in rows:
        missing = REQUIRED_CASE_KEYS - set(row)
        if missing:
            raise ValueError(f"{path} row {row_index} missing keys: {sorted(missing)}")
        agents = row["agents"]
        query = row["user query"]
        if not isinstance(agents, list) or not all(isinstance(agent, dict) for agent in agents):
            raise ValueError(f"{path} row {row_index}: agents must be a list of objects")
        if not isinstance(query, str):
            raise ValueError(f"{path} row {row_index}: user query must be a string")
        for agent in agents:
            if set(agent) != {"agent_name", "agent_description"}:
                field_shapes["agent_key_variation"] += 1
            if not isinstance(agent.get("agent_name"), str) or not isinstance(agent.get("agent_description"), str):
                raise ValueError(f"{path} row {row_index}: malformed agent object")

        source = path.relative_to(repo_root).as_posix()
        case_id = _case_id(source, row_index)
        # Deliberately omit agent_description: compromised descriptions are
        # evaluator-controlled system prompts, not an observed runtime event.
        public_records.append(
            {
                "case_id": case_id,
                "domain": domain,
                "agent_roster": [agent["agent_name"] for agent in agents],
                "user_query": query,
                "public_input_policy": "role_names_and_user_query_only; no_attack_label",
            }
        )
        label_records.append(
            {
                "case_id": case_id,
                "is_adversarial": 1,
                "attack_type": attack,
                "source_file": source,
                "source_row": row_index,
                "label_policy": "evaluator_only; never a monitor feature",
            }
        )
        attack_counts[attack] += 1
        domain_counts[domain] += 1
        agent_count[len(agents)] += 1

    with public_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in public_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with labels_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in label_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema_version": "tamas-static-boundary-v1",
        "source": "https://github.com/microsoft/TAMAS",
        "source_commit": "resolved_at_audit_time",
        "code_license": "MIT",
        "data_license": "Community Data License Agreement - Permissive 2.0",
        "case_count": len(public_records),
        "attack_counts": dict(sorted(attack_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "agent_count_histogram": {str(k): v for k, v in sorted(agent_count.items())},
        "static_release_has_runtime_traces": False,
        "static_release_has_benign_case_files": False,
        "public_fields": ["case_id", "domain", "agent_roster", "user_query"],
        "evaluator_fields": ["is_adversarial", "attack_type", "source_file", "source_row"],
        "label_feature_separation": "labels.jsonl is evaluator-only and must not be passed to feature extraction",
        "api_only_readiness": "requires official AutoGen/CrewAI execution to produce observable runtime logs before scoring",
        "notes": [
            "The checked-in data directory contains 300 attack cases (50 per attack type), not the 100 harmless tasks claimed in the paper abstract/README.",
            "No runtime messages, tool calls, tool results, timestamps, permissions, or terminal outcomes are included in the static JSON cases.",
            "Agent descriptions can contain attack instructions; this projection omits them from monitor-visible public records to avoid configuration-label leakage.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official TAMAS release without label leakage")
    parser.add_argument("--repo", type=Path, required=True, help="root of a local clone of microsoft/TAMAS")
    parser.add_argument("--output", type=Path, required=True, help="output directory for the separated audit projection")
    args = parser.parse_args()
    manifest = audit(args.repo, args.output)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
