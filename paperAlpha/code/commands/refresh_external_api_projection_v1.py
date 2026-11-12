"""Refresh public-only projections for the two frozen API cohorts.

This is a mechanical projection helper. It never writes labels into the public
episode file; reference labels are kept in the paired evaluator file.
"""
from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path


def load_converter():
    here = Path(__file__).resolve()
    spec = importlib.util.spec_from_file_location("external_eval_projection", here.with_name("evaluate_external_api_cohort_v1.py"))
    if spec is None or spec.loader is None:
        raise ImportError("external evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refresh(source: Path, output: Path):
    converter = load_converter()
    episodes, metadata = [], []
    for run_dir in sorted((source / "runs").iterdir()):
        if not run_dir.is_dir() or not (run_dir / "RUN.json").exists():
            continue
        run = json.loads((run_dir / "RUN.json").read_text(encoding="utf-8"))
        if run.get("status") != "completed":
            continue
        episode, label, meta = converter.make_episode(run_dir)
        episodes.append(episode)
        metadata.append(meta)
    if not episodes or len({m["label"] for m in metadata}) < 2:
        raise RuntimeError(f"incomplete or single-class cohort: {source}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes_public.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in episodes) + "\n", encoding="utf-8")
    (output / "labels_reference.jsonl").write_text("\n".join(json.dumps({"episode_id": m["episode_id"], "label": m["label"], "scenario_family": source.name}, ensure_ascii=False) for m in metadata) + "\n", encoding="utf-8")
    print(json.dumps({"source": str(source), "output": str(output), "episodes": len(episodes), "positives": sum(m["label"] for m in metadata)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    args = parser.parse_args()
    if len(args.source) != len(args.output):
        raise SystemExit("each --source needs one matching --output")
    for source, output in zip(args.source, args.output):
        refresh(source.resolve(), output.resolve())
