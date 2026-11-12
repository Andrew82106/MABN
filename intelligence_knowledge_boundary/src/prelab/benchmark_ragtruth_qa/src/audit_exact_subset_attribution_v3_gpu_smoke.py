"""Independent read-only audit of the v3 pilot GPU smoke artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/exact_subset_attribution_v3_fit_pilot"
V2 = ROOT / "results/exact_subset_attribution_v2"
RUNNER = ROOT / "src/run_exact_subset_attribution_v3_fit_pilot.py"
GATE = ROOT / "src/run_exact_subset_attribution_v2.py"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def main():
    smoke = read(OUT / "GPU_SMOKE.json")
    freeze = read(OUT / "design_freeze.json")
    preparation = read(OUT / "preparation_complete.json")
    selection = read(OUT / "selection_manifest.json")
    v2_protocol = read(V2 / "protocol.json")
    assert smoke["status"] == "passed_not_full_extraction"
    signature = smoke["runtime_signature"]
    assert signature["source_sha256"] == sha(RUNNER) == freeze["source_sha256"]
    assert signature["v2_gate_source_sha256"] == sha(GATE)
    assert signature["protocol_sha256"] == freeze["protocol_sha256"]
    assert signature["selection_manifest_sha256"] == preparation["files_sha256"]["selection_manifest.json"]
    assert signature["selected_prepared_sha256"] == preparation["selected_prepared_sha256"]
    lease = smoke["gate_lease"]
    assert lease["consumer_version"] == signature["version"]
    assert lease["v2_gate_source_sha256"] == signature["v2_gate_source_sha256"]
    assert lease["stage"].endswith(":gpu-smoke")
    assert lease["free_bytes_before_load"] >= 6 * 1024 ** 3
    allowlist = set(v2_protocol["wddm_gpu_gate"]["gui_allowlist"])
    own_pid = lease["pid"]
    scan_counts = []
    for key in ("foreign_scan_before", "foreign_scan_after_cuda_init"):
        scan = lease[key]
        assert scan["blockers"] == []
        assert scan["query_row_count"] == len(scan["decisions"])
        for item in scan["decisions"]:
            assert item["decision"] == "allow"
            if item["pid"] == own_pid:
                assert item["reason"] == "own_runner_pid"
            else:
                assert item["reason"] == "explicit_na_reviewed_wddm_graphics"
                assert item["memory_raw"] == "[N/A]"
                assert item["memory_kind"] == "explicit_bracketed_na"
                assert item["nvidia_type"] in {"G", "C+G"}
                assert item["process_basename"] in allowlist
        scan_counts.append(scan["query_row_count"])
    assert smoke["forward_calls"] == 17
    assert smoke["full_view_repeat_max_abs"] <= 1e-6
    assert len(smoke["records"]) == 2
    assert [item["response_id"] for item in smoke["records"]] == [
        selection["selected"][0]["response_id"],
        selection["selected"][-1]["response_id"],
    ]
    assert max(item["shapley_efficiency_max_abs"] for item in smoke["records"]) <= 1e-12
    model = smoke["model"]
    assert model["architecture"] == {
        "model_type": "llama", "layers": 32, "heads": 32,
        "kv_heads": 32, "head_dim": 128, "hidden_size": 4096,
        "lb_width": 1024, "vocab_size": 32000,
    }
    assert model["attention_backend"] == "sdpa"
    assert smoke["fit_gold_files_opened"] is False
    assert smoke["labels_accessed"] is False
    assert smoke["official_test_opened"] is False
    result = {
        "status": "passed_independent_read_only_GPU_smoke_audit",
        "audit_source_sha256": sha(__file__),
        "GPU_SMOKE_sha256": sha(OUT / "GPU_SMOKE.json"),
        "runtime_signature_sha256": smoke["runtime_signature_sha256"],
        "checks": {
            "runner_gate_protocol_selection_hashes_exact": True,
            "strict_WDDM_exception_replayed_for_every_foreign_entry": True,
            "no_gate_blocker": True,
            "free_memory_at_least_6GiB": True,
            "same_Llama2_7B_32layer_SDPA_backend": True,
            "first_last_hash_selected_rows_used": True,
            "repeat_exact": smoke["full_view_repeat_max_abs"] == 0.0,
            "Shapley_efficiency_within_1e_12": True,
        },
        "measurements": {
            "scan_row_counts_before_after": scan_counts,
            "free_bytes_before_load": lease["free_bytes_before_load"],
            "allocated_after_load_bytes": model["allocated_after_load_bytes"],
            "model_load_seconds": model["load_seconds"],
            "smoke_compute_seconds": smoke["seconds_after_model_load"],
            "forward_calls": smoke["forward_calls"],
        },
        "fit_gold_files_opened": False,
        "labels_accessed": False,
        "official_test_opened": False,
        "paper_baseline_modified_or_scored": False,
    }
    save(OUT / "GPU_SMOKE_AUDIT.json", result)
    print("EXACT_SUBSET_V3_GPU_SMOKE_INDEPENDENT_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    main()
