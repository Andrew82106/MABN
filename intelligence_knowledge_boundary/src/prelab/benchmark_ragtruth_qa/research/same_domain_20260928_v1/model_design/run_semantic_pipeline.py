"""Root-authorized serial GPU pipeline, after extractor PID89944 releases GPU."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATUS = HERE / "PIPELINE_STATUS.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def status(stage, **extra):
    save(STATUS, {"pipeline_pid": os.getpid(), "stage": stage, "updated_at": time.time(),
                  "test_labels_read": False, "test_inference_run": False, **extra})


def execute(stage, runner, *arguments):
    command = [sys.executable, "-u", "-B", "-X", "utf8", str(HERE / runner), *map(str, arguments)]
    log_path = HERE / f"pipeline_{stage}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(HERE), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        status(stage, child_pid=process.pid, command=command, log=str(log_path))
        print("STAGE_STARTED", stage, "launcher", process.pid, flush=True)
        tree_saved = False
        for line in process.stdout:
            log.write(line); log.flush()
            print(stage, line.rstrip(), flush=True)
            if not tree_saved:
                children = psutil.Process(process.pid).children(recursive=True)
                status(stage, child_pid=process.pid, model_pids=[p.pid for p in children],
                       command=command, log=str(log_path))
                tree_saved = bool(children)
        code = process.wait()
        if code:
            raise RuntimeError(f"{stage} failed with exit code {code}; no fallback or retry")
    print("STAGE_EXITED", stage, code, flush=True)


def main():
    completion = ROOT / "signals/COMPLETE_fit_validation.json"
    print("PIPELINE_STARTED", os.getpid(), flush=True)
    while True:
        record = read(completion) if completion.exists() else {}
        alive = psutil.pid_exists(89944)
        ready = record.get("count") == record.get("expected_count") == 1563
        status("waiting_for_extractor", extractor_pid=89944, extractor_alive=alive,
               complete_count=record.get("count"), expected_count=record.get("expected_count"))
        if ready and not alive:
            break
        time.sleep(50)
    print("EXTRACTOR_COMPLETE_AND_EXITED", flush=True)
    large = HERE / "modernbert_large_v1"
    execute("large_smoke", "train_full_context.py", "gpu-smoke", "--out", large)
    execute("large_train", "train_full_context.py", "train", "--out", large)
    candidates = {name: read(HERE / f"modernbert_{name}_v1/complete.json") for name in ("base", "large")}
    keys = {name: rec["best"]["selection_key"][:3] for name, rec in candidates.items()}
    chosen = "large" if tuple(keys["large"]) > tuple(keys["base"]) else "base"
    full_run = HERE / f"modernbert_{chosen}_v1"
    selection = {"chosen": chosen, "keys": keys,
        "rule": "existing validation no_hint min(two F1), windowF1, windowPrecision; exact architecture tie chooses smaller base",
        "fixed_oof_epochs": candidates[chosen]["best"]["epoch"],
        "source_complete_sha256": {name: sha(HERE / f"modernbert_{name}_v1/complete.json") for name in candidates},
        "test_labels_read": False}
    save(HERE / "SEMANTIC_SELECTION.json", selection)
    print("SEMANTIC_SELECTED", json.dumps(selection), flush=True)
    outputs = []
    for fold in range(3):
        out = HERE / (f"modernbert_oof_fold_{fold}" if chosen == "base" else f"modernbert_large_oof_fold_{fold}")
        ids = ROOT / f"fusion/folds/fold_{fold}_fit_ids.json"
        if not (out / "prepared.json").exists():
            execute(f"fold_{fold}_prepare", "train_crossfit.py", "prepare", "--full-run", full_run,
                    "--fit-ids", ids, "--out", out)
        assert read(out / "prepared.json")["protocol"]["epochs"] == selection["fixed_oof_epochs"]
        assert Path(read(out / "prepared.json")["protocol"]["model"]) == Path(read(full_run / "prepared.json")["protocol"]["model"])
        execute(f"fold_{fold}_cpu", "train_crossfit.py", "cpu-check", "--out", out)
        execute(f"fold_{fold}_train", "train_crossfit.py", "train", "--out", out)
        manifest = out / "held_prediction_manifest.json"
        assert manifest.exists()
        outputs.append({"fold": fold, "manifest": str(manifest), "manifest_sha256": sha(manifest)})
    expected, records = set(), {}
    for fold in range(3):
        held = read(ROOT / f"fusion/folds/fold_{fold}_held_ids.json")
        assert not expected & set(held)
        expected.update(held)
        manifest = read(outputs[fold]["manifest"])
        prediction_path = Path(manifest["predictions"]["path"])
        assert sha(prediction_path) == manifest["predictions"]["sha256"]
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            assert record["response_id"] not in records
            records[record["response_id"]] = record
    assert set(records) == expected and len(records) == 1292
    aggregate = HERE / "selected_semantic_oof_predictions.jsonl"
    with aggregate.open("w", encoding="utf-8") as stream:
        for rid in sorted(records):
            stream.write(json.dumps(records[rid], ensure_ascii=False) + "\n")
    result = {"selection": selection, "fold_manifests": outputs, "oof_answers": len(records),
        "aggregate_predictions": str(aggregate), "aggregate_sha256": sha(aggregate),
        "all_child_processes_exited": True, "GPU_released": True,
        "test_labels_read": False, "test_inference_run": False}
    save(HERE / "PIPELINE_COMPLETE.json", result)
    status("complete", **result)
    print("PIPELINE_COMPLETE_GPU_RELEASED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        status("failed", traceback=failure, no_automatic_fallback=True)
        print(failure, file=sys.stderr, flush=True)
        raise
