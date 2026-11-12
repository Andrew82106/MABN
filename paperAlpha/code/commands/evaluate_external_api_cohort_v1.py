"""Evaluate the frozen two-line monitor on an independently collected API MAS cohort.

The cohort is the 8-case three-agent authority pilot.  The reference-policy
labels are read only from post-run evaluator manifests; the public episode
projection deliberately excludes those fields and is the only input given to
the monitor.  This is an external transfer check, not a refit or a benchmark
claim: the cohort is small and all cases use the same local API gateway.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / 'experiments/synthetic_mas/runs/independent_mas_v3'
API_ROOT = ROOT / 'experiments/delegation/runs/branched_authority_api_20260919'


def load_main():
    spec = importlib.util.spec_from_file_location("journal_eval", Path(__file__).with_name("evaluate_independent_mas_journal_v1.py"))
    if spec is None or spec.loader is None:
        raise ImportError("journal evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_two():
    spec = importlib.util.spec_from_file_location("two_layer_external", Path(__file__).with_name("evaluate_two_layer_v2.py"))
    if spec is None or spec.loader is None:
        raise ImportError("two layer evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_episode(run_dir: Path) -> tuple[dict, int, dict]:
    run = json.loads((run_dir / "RUN.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    raw = read_jsonl(run_dir / "monitor_events.jsonl")
    events = [{"event_type": "input_received", "actor": "planner", "data_class": "untrusted", "sequence": 0}]
    for item in raw:
        if item.get("event_type") == "message_sent":
            # Keep the original observable class.  Unknown provenance is an
            # uncertainty signal, never silently treated as trusted.
            events.append({k: item[k] for k in ("event_type", "actor", "target", "content", "data_class", "sequence") if k in item})
        elif item.get("event_type") == "tool_call":
            # Map the local simulator's harmless payment name to the public
            # privileged-tool vocabulary used by the frozen monitor.
            event = {k: item[k] for k in ("event_type", "actor", "principal", "arguments", "grant_ids", "sequence") if k in item}
            event.update({"tool": "payments.write", "data_class": "unknown"})
            events.append(event)
        elif item.get("kind") == "grant":
            # Authority records are public evidence in this cohort. Preserve
            # the scope/lineage fields for the knowledge-compiled binding
            # extension, and separately expose the communication edge used by
            # the frozen runtime line. These are not evaluator labels.
            events.append({"kind": "grant", "payload": item.get("payload", {}),
                           "observer": item.get("observer"), "authenticated_actor": item.get("authenticated_actor"),
                           "sequence": item.get("sequence"), "event_id": item.get("event_id")})
            payload = item.get("payload", {})
            if payload.get("parent_id") is not None:
                events.append({"event_type": "delegation", "actor": payload.get("issuer"), "target": payload.get("recipient"),
                               "data_class": "unknown", "edge_kind": "workflow", "edge_confidence": 1.0,
                               "sequence": item.get("sequence")})
        elif item.get("kind") == "revoke":
            # Revocation is public state; its presence is retained, but no
            # policy outcome or permission-mismatch flag is injected.
            events.append({"kind": "revoke", "payload": item.get("payload", {}),
                           "observer": item.get("observer"), "authenticated_actor": item.get("authenticated_actor"),
                           "event_type": "authority_update", "actor": "user", "data_class": "unknown",
                           "sequence": item.get("sequence"), "event_id": item.get("event_id")})
    public = {"episode_id": config["id"], "agents": list(config["workflow"]["nodes"]),
              "workflow_dag": {"nodes": list(config["workflow"]["nodes"]), "edges": [list(e) for e in config["workflow"]["edges"]]},
              "events": events}
    label = int(run.get("violations", 0) > 0)
    meta = {"episode_id": config["id"], "layout": config.get("layout"), "treatment": config.get("treatment"),
            "label": label, "api_calls": run.get("API_calls"), "violations": run.get("violations"),
            "source": str(run_dir)}
    return public, label, meta


def metrics(y, p, threshold=.5):
    pred = p >= threshold
    return {"n": int(len(y)), "positive": int(y.sum()), "threshold": float(threshold),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 and len(np.unique(p)) > 1 else None,
            "auprc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "predicted_positive": int(pred.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-root", type=Path, default=API_ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / 'experiments/delegation/runs/submission__external_api_mas_v1')
    args = ap.parse_args()
    journal = load_main(); two = load_two()
    train_traces = journal.read_jsonl(MAIN / "traces_public.jsonl")
    train_labels = {r["episode_id"]: r for r in journal.read_jsonl(MAIN / "labels.jsonl")}
    api_root = args.api_root.resolve()
    api_dirs = sorted((api_root / "runs").iterdir())
    episodes, labels, metadata = [], [], []
    for run_dir in api_dirs:
        if not run_dir.is_dir() or not (run_dir / "RUN.json").exists():
            continue
        run = json.loads((run_dir / "RUN.json").read_text(encoding="utf-8"))
        if run.get("status") != "completed":
            continue
        ep, label, meta = make_episode(run_dir)
        episodes.append(ep); labels.append(label); metadata.append(meta)
    if len(episodes) < 4 or len(set(labels)) < 2:
        raise RuntimeError("external cohort lacks completed cases or both labels")
    y_train = np.asarray([int(train_labels[e["episode_id"]]["label"]) for e in train_traces], dtype=int)
    y_api = np.asarray(labels, dtype=int)
    wf_train = np.asarray([journal.workflow_vector(e) for e in train_traces]); rt_train = np.asarray([journal.runtime_vector(e) for e in train_traces])
    wf_api = np.asarray([journal.workflow_vector(e) for e in episodes]); rt_api = np.asarray([journal.runtime_vector(e) for e in episodes])
    # Fit once on the independent synthetic development set.  This is a
    # transfer model, not a refit: no API label is used for fitting,
    # thresholding or tuning.  The journal headline remains the nested
    # grouped-OOF result in independent_mas_journal_v1; this external check
    # is intentionally kept computationally small and descriptive.
    wf_bn = journal.workflow_bn_fit(wf_train, y_train)
    p_wf_train = journal.workflow_bn_score(wf_train, wf_bn)
    p_wf_api = journal.workflow_bn_score(wf_api, wf_bn)
    _, p_wf_log_api, _ = journal.fit_logistic(wf_train, y_train, wf_api, 301)
    _, p_rt_log_api, _ = journal.fit_logistic(rt_train, y_train, rt_api, 302)
    train_eps = [{**train_traces[i], "label": int(y_train[i])} for i in range(len(train_traces))]
    bn = two.fit_bn(train_eps, y_train)
    theta, opt = two.fit_model(train_eps, y_train, bn, 401)
    p_rt_train = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in train_eps])
    p_rt_api = np.asarray([two._propagate(theta, two.event_features(e), bn) for e in episodes])
    fusion_train = np.c_[wf_train, rt_train, p_wf_train, p_rt_train]
    fusion_api = np.c_[wf_api, rt_api, p_wf_api, p_rt_api]
    fusion = LogisticRegression(C=1.0, max_iter=1000, random_state=501).fit(fusion_train, y_train)
    scores = {"workflow_bn": p_wf_api, "runtime_bn": p_rt_api,
              "workflow_logistic": p_wf_log_api, "runtime_logistic": p_rt_log_api,
              "hierarchical_fusion": fusion.predict_proba(fusion_api)[:, 1]}
    result = {k: metrics(y_api, v) for k, v in scores.items()}
    acquisition_report = api_root / "REPORT.json"
    acquisition = json.loads(acquisition_report.read_text(encoding="utf-8")) if acquisition_report.exists() else {}
    generation_endpoint = acquisition.get("generation_endpoint", "unknown")
    second_endpoint = "lanyun.net" in str(generation_endpoint).lower()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "episodes_public.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in episodes) + "\n", encoding="utf-8")
    (args.out / "labels_reference.jsonl").write_text("\n".join(json.dumps({"episode_id": m["episode_id"], "label": m["label"], "scenario_family": "branched_authority_api"}, ensure_ascii=False) for m in metadata) + "\n", encoding="utf-8")
    (args.out / "predictions.jsonl").write_text("\n".join(json.dumps({"episode_id": m["episode_id"], "label": m["label"], **{k: float(scores[k][i]) for k in scores}}, sort_keys=True) for i, m in enumerate(metadata)) + "\n", encoding="utf-8")
    report = {"schema": "external-api-mas-transfer-v1", "source": str(api_root), "n": len(episodes), "positive": int(y_api.sum()),
              "generation_endpoint": generation_endpoint, "second_generation_endpoint": second_endpoint,
              "fit_source": str(MAIN), "api_labels_used_for_fit": False,
              "cohort_note": "Completed real-API decisions in a hand-configured simulated authority environment; descriptive transfer check, not independent deployment or journal headline.",
              "runtime_optimizer": {"success": bool(opt.get("success", False)), "iterations": int(opt.get("iterations", -1))},
              "metrics": result, "cases": metadata}
    (args.out / "REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# External API MAS transfer check", "", f"Completed API cases: **{len(episodes)}**; reference-policy positives: **{int(y_api.sum())}**.", "", "All estimators were fitted on `independent_mas_v3` only; API labels were not used for fitting or threshold selection.", "", "| method | F1 | precision | recall | AUROC | AUPRC | Brier |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in result.items():
        auroc = "NA" if m["auroc"] is None else f"{m['auroc']:.3f}"
        lines.append(f"| {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {auroc} | {m['auprc']:.3f} | {m['brier']:.3f} |")
    boundary = ("This is a small second-endpoint API transfer check." if second_endpoint
                else "This is a small same-gateway API transfer check.")
    lines += ["", "## Claim boundary", boundary + " It is not a statistically powered MAS benchmark and still requires a larger independently adjudicated cohort before being used as a journal headline.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
