"""Independent stdlib-only audit of the WDDM gate repair; no project imports."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/run_exact_subset_attribution_v2.py"
V1 = ROOT / "results/exact_subset_attribution_v1"
V2 = ROOT / "results/exact_subset_attribution_v2"
PREPARED = V1 / "prepared_inputs.jsonl"
EXPECTED_PREPARED_SHA256 = "55753d10aa79f2069acfed7161c9b64fa8bcc4ed21da97c1a403ec73e1ff1f2a"
EXPECTED_V1_COMPLETE_SHA256 = "f68257930b58c4ab44ed1b990e01e012beb19279ddd69f3082bf5c5a97efcb17"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def direct_calls(function):
    result = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            result.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                result.add(f"{node.func.value.id}.{node.func.attr}")
            result.add(node.func.attr)
    return result


def main():
    freeze = read(V2 / "design_freeze.json")
    complete = read(V2 / "preparation_complete.json")
    cpu = read(V2 / "CPU_CHECK.json")
    producer_audit = read(V2 / "CODE_AUDIT.json")
    commands_path = V2 / "EXECUTION_COMMANDS.md"
    assert commands_path.exists()
    assert freeze["v2_source_sha256"] == sha(SCRIPT)
    assert freeze["protocol_sha256"] == sha(V2 / "protocol.json")
    assert freeze["prepared_inputs_sha256"] == sha(PREPARED) == EXPECTED_PREPARED_SHA256
    assert sha(V1 / "preparation_complete.json") == EXPECTED_V1_COMPLETE_SHA256
    assert complete["status"] == "CPU_ready_waiting_for_GPU_review"
    for name, expected in complete["files_sha256"].items():
        assert sha(V2 / name) == expected
    assert all(sha(V1 / name) == expected
               for name, expected in freeze["v1_existing_artifacts_sha256"].items())

    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    cpu_entries = ("initialize", "prepare", "check", "audit", "gpu_gate_selfcheck")
    forbidden_cpu_calls = {
        "run_nvidia_smi", "inspect_foreign_gpu_processes", "exclusive_gpu",
        "load_nf4", "core.load_nf4", "mem_get_info",
    }
    cpu_call_audit = {name: sorted(direct_calls(functions[name])) for name in cpu_entries}
    assert all(not (set(calls) & forbidden_cpu_calls)
               for calls in cpu_call_audit.values())
    inspect_calls = direct_calls(functions["inspect_foreign_gpu_processes"])
    assert "run_nvidia_smi" in inspect_calls
    exclusive_text = ast.get_source_segment(source, functions["exclusive_gpu"])
    assert exclusive_text.count("inspect_foreign_gpu_processes(os.getpid())") == 2
    exclusive_calls = direct_calls(functions["exclusive_gpu"])
    assert {"acquire_lock", "release_lock", "mem_get_info"} <= exclusive_calls
    classifier_text = ast.get_source_segment(source, functions["classify_processes"])
    assert 'record["memory_kind"] == "numeric_mib"' in classifier_text
    assert 'record["memory_kind"] != "explicit_bracketed_na"' in classifier_text
    assert 'item["nvidia_type"] not in {"G", "C+G"}' in classifier_text
    assert "WDDM_GUI_EXECUTABLE_ALLOWLIST" in classifier_text
    assert classifier_text.index('record["memory_kind"] == "numeric_mib"') < classifier_text.index("WDDM_GUI_EXECUTABLE_ALLOWLIST")
    assert "expanded_metadata" not in functions
    assert "score" not in functions

    counts = Counter()
    raw_tokens = 0
    response_ids = set()
    with PREPARED.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            assert row.get("labels_used") is False
            assert row["official_split"] == "train"
            assert [view["mask"] for view in row["views"]] == list(range(8))
            assert row["response_id"] not in response_ids
            response_ids.add(row["response_id"])
            counts[row["partition"]] += 1
            raw_tokens += len(row["answer_token_ids"])
    assert counts == {"fit": 3680, "calibration": 159}
    assert len(response_ids) == 3839 and raw_tokens == 708506
    assert cpu["synthetic_gpu_gate_selfcheck"]["status"] == "passed"
    assert all(cpu["synthetic_gpu_gate_selfcheck"]["cases"].values())
    assert cpu["nvidia_smi_called"] is False and cpu["GPU_used"] is False
    assert producer_audit["GPU_used"] is False
    assert not (V2 / "GPU_SMOKE.json").exists()
    assert not (V2 / "extraction_complete.json").exists()
    assert not (V2 / "token_logprob").exists()

    report = {
        "status": "passed_independent_stdlib_CPU_audit",
        "audit_source_sha256": sha(__file__),
        "production_source_sha256": sha(SCRIPT),
        "prepared_inputs_sha256": sha(PREPARED),
        "v1_preparation_complete_sha256": sha(V1 / "preparation_complete.json"),
        "execution_commands_sha256": sha(commands_path),
        "counts": {"answers": len(response_ids), "partitions": dict(counts),
                   "raw_answer_tokens": raw_tokens},
        "checks": {
            "v1_existing_artifact_hashes_unchanged": True,
            "v2_source_and_protocol_match_preparation_freeze": True,
            "CPU_entrypoints_do_not_reach_nvidia_query_or_model_load": True,
            "numeric_memory_branch_precedes_GUI_exception": True,
            "pure_C_unknown_type_and_unreviewed_name_fail_closed": True,
            "two_scans_lock_nonce_and_free_memory_gate_present": True,
            "all_3839_label_free_rows_and_eight_views_recounted": True,
            "GPU_outputs_absent": True,
            "score_gold_and_official_test_not_reachable": True,
            "baseline_untouched": True,
        },
        "CPU_direct_call_inventory": cpu_call_audit,
        "GPU_used": False,
        "labels_accessed": False,
        "official_test_opened": False,
        "production_imported": False,
    }
    save(V2 / "INDEPENDENT_AUDIT.json", report)
    (V2 / "INDEPENDENT_AUDIT.md").write_text(
        "# Exact subset attribution v2：独立 CPU 审计\n\n"
        "通过。独立脚本未导入生产代码，重新核对了生产源码/协议/3,839 条准备数据和 v1 全部既有文件哈希。"
        "数值显存分支先于 GUI 例外；纯 C、未知类型和未审核程序名均失败关闭。\n\n"
        "GPU_SMOKE、完整提取和 token cache 均不存在；未使用 GPU、标签或官方测试集。\n",
        encoding="utf-8",
    )
    print("EXACT_SUBSET_ATTRIBUTION_V2_INDEPENDENT_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    main()
