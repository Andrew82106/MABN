"""Export auditable conditional BN instances from a saved development run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
from paperalpha_monitor.streaming_bn import CELL_NAMES, PRIVILEGED_TOOLS, replay  # noqa: E402
from paperalpha_monitor.template_registry import load_registry  # noqa: E402


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / 'experiments/runtime_alarm/runs/prefix_forecast_v1_20260919')
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--out", type=Path, default=ROOT / 'experiments/model_mechanisms/runs/template_graph_examples_20260919')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite graph examples")
    manifest = read(args.run / "manifest.json")
    trace_path = ROOT / manifest["config"]["data"] / "traces_public.jsonl"
    expected_sha = manifest["input_sha256"][str(trace_path.relative_to(ROOT)).replace("\\", "/")]
    if sha(trace_path) != expected_sha:
        raise ValueError("trace data differs from saved experiment")
    run_dir = args.run / "workflow_known__observed" / str(args.seed)
    parameter_path = run_dir / "model_parameters.json"
    parameters = read(parameter_path)["prefix_bn"]["bn"]
    if parameters["cpt_cell_names"] != CELL_NAMES:
        raise ValueError("saved parameter schema differs from registry")
    theta = np.asarray(parameters["theta"])
    test_ids = set(manifest["splits"][str(args.seed)]["test"]["episode_ids"])
    candidates = [trace for trace in jsonl(trace_path) if trace["episode_id"] in test_ids]
    # Explanation sampling uses only observed factors, never outcome labels.
    chosen = None
    for trace in candidates:
        trajectory = replay(trace, workflow=trace["workflow_dag"], permission_mode="observed")
        if chosen is None:
            chosen = trace, trajectory
        if len({factor["template"] for factor in trajectory[-1].factors}) == 4:
            chosen = trace, trajectory
            break
    if chosen is None:
        raise ValueError("no saved test trace available")
    trace, trajectory = chosen
    saved = next(row for row in jsonl(run_dir / "predictions.jsonl")
                 if row["split"] == "test" and row["episode_id"] == trace["episode_id"])
    registry = load_registry()
    compiled = [registry.compile_snapshot(snapshot) for snapshot in trajectory]
    probabilities = [graph.risk_probability(theta) for graph in compiled]
    np.testing.assert_allclose(probabilities, saved["prefix_scores"]["prefix_bn"], rtol=1e-12, atol=1e-12)
    first_dispatch = next((i for i, event in enumerate(trace["events"])
                           if event.get("event_type") == "tool_call" and event.get("tool") in PRIVILEGED_TOOLS), None)
    selected = sorted({0, len(trajectory) - 1} | ({max(0, first_dispatch - 1), first_dispatch} if first_dispatch is not None else set()))
    args.out.mkdir(parents=True)
    source_paths = [Path(__file__), ROOT / 'knowledge/templates/risk_templates_v1.json', ROOT / 'knowledge/risks/risk_catalog.json',
                    *(ROOT / 'code/src/paperalpha_monitor').glob("*.py")]
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(ROOT)
        target = args.out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[str(relative)] = sha(target)
    payload = {
        "episode_id": trace["episode_id"], "seed": args.seed, "setting": "workflow_known__observed",
        "selection": "first saved test trace with all four instantiated template types, fallback first; labels not used",
        "scope": "graph compilation and numerical equivalence, not a new fit or independent performance result",
        "training_target": manifest["config"]["forecast_target"],
        "parameter_file": str(parameter_path), "parameter_sha256": sha(parameter_path),
        "input_sha256": sha(trace_path), "source_sha256": source_hashes,
        "registry": registry.semantic_metadata(),
        "parameters": parameters,
        "snapshots": [{"event_index": i, "risk_probability": probabilities[i], "graph": compiled[i].to_dict()} for i in selected],
        "verified_prefix_count": len(trajectory),
        "limitations": ["Observed-parent CPT instance graph, not workflow-role sub-BN discovery",
                        "No learned graph gates or causal interpretation",
                        "Source references validate catalog linkage, not independent verification of every paper claim"],
    }
    (args.out / "graphs.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# 知识模板生成的条件概率图示例", "",
              "这是已有前缀模型的可检查图实例，不是新训练或性能提升。", "",
              f"- 样例任务：{trace['episode_id']}；按可见模板覆盖选取，不按危险标签选取。",
              f"- 全部 {len(trajectory)} 个前缀的图精确边缘概率与已有预测逐一相等。",
              "- 图结构：可观察父变量 → 动作/全局风险 Bernoulli 节点 → 系统 OR。",
              "- 同一动作共享权限观测父节点；同类风险在不同动作间共享概率表。",
              "- 每个实例保留时刻、Agent、动作和真实通信证据；缺少工作流/权限不冒充已确认安全。",
              "- graphs.json 含图节点/边、条件表参数索引、学到的参数、文献关联及其覆盖缺口。",
              "- 原定角色子 BN 和可学习全图层尚未完成，不能用这张导出图替代。", ""]
    (args.out / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output": str(args.out), "episode_id": trace["episode_id"],
                      "verified_prefix_count": len(trajectory), "snapshot_count": len(selected)}))


if __name__ == "__main__":
    main()
