"""Audit whether existing MAS artifacts satisfy the independent confirmation gate.

This script is intentionally read-only with respect to source artifacts.  It
counts frozen files and metadata, records the label/topology boundary, and
emits a conservative report.  It never combines incompatible data sources and
never treats benchmark/oracle labels as independent human adjudication.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_case_dirs(path: Path) -> int:
    cases = path / "cases"
    return sum(item.is_dir() for item in cases.iterdir()) if cases.exists() else 0


def count_collection(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {"case_dirs": count_case_dirs(path)}
    collection = path / "COLLECTION.json"
    if collection.exists():
        data = load_json(collection)
        rows = data.get("cases", [])
        value["collection_rows"] = len(rows)
        value["status_counts"] = dict(Counter(row.get("status") for row in rows))
        value["completed"] = sum(row.get("status") == "completed" for row in rows)
        value["episodes"] = value["completed"]
    return value


def audit(repo: Path) -> dict[str, Any]:
    dev = repo / "paperAlpha" / "results" / "submission" / "development"
    sources: dict[str, Any] = {}

    mast = repo / "paperAlpha" / "data" / "external" / "MAST"
    if (mast / "MAD_full_dataset.json").exists():
        rows = load_json(mast / "MAD_full_dataset.json")
        groups = Counter((row.get("mas_name"), row.get("benchmark_name")) for row in rows)
        sources["mast_mad"] = {
            "full_traces": len(rows),
            "mas_benchmark_groups": len(groups),
            "human_labelled_traces": len(load_json(mast / "MAD_human_labelled_dataset.json")),
            "explicit_topology_families": 0,
            "api_conditions": 0,
            "label_boundary": "MAST annotations and 19-trace human-labelled release; not this protocol's blind double labels",
        }

    a2a = dev / "a2asecbench_api_final_20261002"
    if a2a.exists():
        configs = sorted(a2a.glob("*_config.json"))
        family_rows = {}
        for config in configs:
            data = load_json(config)
            family_rows[config.stem.removesuffix("_config").upper()] = len(data.get("cases", []))
        sources["a2asecbench"] = {
            "episodes": sum(family_rows.values()),
            "attack_families": len(family_rows),
            "family_counts": family_rows,
            "explicit_topology_families": 0,
            "api_conditions": "local endpoint; model varies by family",
            "label_boundary": "benchmark scenario role / executor outcome, not independent human adjudication",
        }

    agentleak = dev / "agentleak_external_journal_v1_r3"
    if (agentleak / "PLAN.json").exists():
        plan = load_json(agentleak / "PLAN.json")
        projection = dev / "agentleak_external_projection_v1" / "coverage.json"
        projection_data = load_json(projection) if projection.exists() else {}
        sources["agentleak"] = {
            "eligible_traces": plan.get("rows"),
            "request_groups": plan.get("request_groups"),
            "positive_reference": plan.get("positive"),
            "explicit_topology_families": 0,
            "api_conditions": 0,
            "label_boundary": "benchmark leak verdict; no blind double-labelled policy contract",
            "projection_trace_count": projection_data.get("trace_count"),
        }

    sources["harnessaudit_initial"] = count_collection(dev / "harnessaudit")
    sources["harnessaudit_expansion"] = count_collection(dev / "harnessaudit_expansion")
    for key in ("harnessaudit_initial", "harnessaudit_expansion"):
        if key in sources:
            sources[key]["explicit_topology_families"] = 1
            sources[key]["api_conditions"] = 1
            sources[key]["label_boundary"] = "author actionwise checker/oracle; no independent double labels"

    for key, directory in {
        "api_lanyun": dev / "balanced_api_mas_transfer_lanyun_r15",
        "api_local": dev / "external_api_mas_transfer_local_20261004",
    }.items():
        report = directory / "REPORT.json"
        if report.exists():
            data = load_json(report)
            sources[key] = {
                "episodes": data.get("n"),
                "positive": data.get("positive"),
                "api_condition": data.get("generation_endpoint"),
                "explicit_topology_families": 1,
                "api_conditions": 1,
                "label_boundary": data.get("cohort_note", "controlled simulator/evaluator labels"),
            }

    topology_report = dev / "unified_topology_api_20261005_lanyun_r2" / "REPORT.json"
    if topology_report.exists():
        data = load_json(topology_report)
        plan = data.get("plan", {})
        sources["unified_topology_smoke"] = {
            "episodes": len(data.get("episodes", [])),
            "topology_families": len(plan.get("topologies", {})),
            "api_conditions": 1,
            "label_boundary": "evaluator-only policy-intent score; not a human security label",
        }

    # Conservative gate: do not sum rows across incompatible sources.
    gate = {
        "minimum_complete_episodes": False,
        "four_explicit_topology_families": False,
        "two_independent_api_model_conditions": False,
        "two_blind_human_labelers": False,
        "single_compatible_frozen_cohort": False,
        "decision": "not_satisfied",
    }
    notes = [
        "Existing counts are retained as source-specific diagnostics; they are not summed into one confirmation set.",
        "A2ASecBench attack families and MAST MAS×benchmark groups are not interchangeable with workflow topology families.",
        "LANYUN/local API runs share a hand-authored simulator and evaluator labels; they are cross-endpoint diagnostics, not independent human confirmation.",
        "The 19 MAST human-labelled traces are too few and outside the frozen API protocol.",
    ]
    return {
        "schema": "paperalpha-independent-confirmation-audit-v1",
        "as_of": "2026-10-05",
        "sources": sources,
        "gate": gate,
        "notes": notes,
        "minimum_freeze": {
            "target_complete_episodes": 160,
            "topologies": ["chain", "fork", "join", "review"],
            "api_model_conditions": 2,
            "per_topology_per_condition": 20,
            "balanced_labels_per_cell": {"benign": 10, "violating": 10},
            "labelers": 2,
            "failures": "retain in attempt denominator; never outcome-filter or selectively replace",
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Independent confirmation source audit",
        "",
        f"As of {report['as_of']}. Decision: **{report['gate']['decision']}**.",
        "",
        "This report is conservative: incompatible datasets are not pooled and benchmark/oracle labels are not renamed as blind human labels.",
        "",
        "| Source | Episodes/traces | Topology/condition evidence | Label boundary |",
        "|---|---:|---|---|",
    ]
    for name, data in report["sources"].items():
        count = data.get("episodes", data.get("full_traces", data.get("eligible_traces", "n/a")))
        topologies = data.get("topology_families", data.get("explicit_topology_families", "n/a"))
        condition = data.get("api_conditions", data.get("api_condition", "n/a"))
        label = data.get("label_boundary", "not recorded")
        lines.append(f"| `{name}` | {count} | topologies={topologies}; conditions={condition} | {label} |")
    lines.extend([
        "",
        "## Gate",
        "",
        "All confirmation-gate fields are false until a single frozen cohort is collected and independently double-labelled.",
        "",
        "## Minimum freeze",
        "",
        "- 160 complete episodes: chain/fork/join/review × two API/model conditions × 20 per cell.",
        "- Each cell has 10 benign and 10 violating cases declared before collection.",
        "- Two blind labelers record violation, first event, class, evidence IDs and rationale; report Cohen's κ.",
        "- Retain failures and never use confirmation outcomes to replace cases or tune the monitor.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paperAlpha/results/submission/development/independent_confirmation_audit_20261005"),
    )
    args = parser.parse_args()
    report = audit(args.repo.resolve())
    out = args.out if args.out.is_absolute() else args.repo.resolve() / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["gate"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
