"""Wait for complete frozen signals, then run CPU full-fit and grouped OOF jobs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
RUN = OUT.parent
SIGNALS = RUN / "signals"
TRAINER = RUN / "train_whitebox.py"
LOGS = OUT / "pipeline_logs"
LOGS.mkdir(exist_ok=True)
STARTED = time.time()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = Path(path).with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def status(stage, **extra):
    record = {"stage": stage, "pipeline_pid": os.getpid(), "started_at": STARTED,
              "updated_at": time.time(), "GPU_used": False, "test_labels_read": False, **extra}
    save(OUT / "PIPELINE_STATE.json", record)
    return record


def wait_signals():
    expected = {}
    for split in ("fit", "validation"):
        with (RUN / "data" / split / "geometry.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                expected[row["response_id"]] = (split, row["input_sha256"])
    assert len(expected) == 1563
    completion = SIGNALS / "COMPLETE_fit_validation.json"
    print(json.dumps(status("waiting_for_signals", expected=1563)), flush=True)
    while not completion.exists():
        progress = read(SIGNALS / "PROGRESS.json") if (SIGNALS / "PROGRESS.json").exists() else {}
        status("waiting_for_signals", expected=1563, extraction_progress=progress)
        time.sleep(50)
    done = read(completion)
    assert done["count"] == done["expected_count"] == 1563, done
    for rid, (split, input_sha) in expected.items():
        arrays, receipt_path = SIGNALS / (rid + ".npz"), SIGNALS / (rid + ".json")
        assert arrays.is_file() and arrays.stat().st_size > 0 and receipt_path.is_file(), rid
        receipt = read(receipt_path)
        assert receipt["response_id"] == rid and receipt["partition"] == split, rid
        assert receipt["input_sha256"] == input_sha, rid
        assert receipt["source_sha256"] == done["source_sha256"], rid
    gate = {"cache_records": len(expected), "completion": done,
            "completion_sha256": sha(completion), "validated_all_cache_paths_and_receipts": True,
            "npz_bytes_sha256_checked_by_trainer_on_first_load": True,
            "test_labels_read": False, "GPU_used": False}
    save(OUT / "CACHE_GATE.json", gate)
    print(json.dumps(status("signals_ready", expected=1563)), flush=True)


def execute(stage, arguments):
    log_path = LOGS / f"{stage}_{int(time.time())}.log"
    command = [sys.executable, "-u", str(TRAINER), *map(str, arguments)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4")
    with log_path.open("w", encoding="utf-8") as handle:
        child = subprocess.Popen(command, cwd=RUN, stdout=handle, stderr=subprocess.STDOUT,
                                 env=environment)
        print(json.dumps(status(stage, child_pid=child.pid, log=str(log_path), command=command)), flush=True)
        try:
            while child.poll() is None:
                time.sleep(50)
        except BaseException:
            child.terminate()
            child.wait(timeout=30)
            raise
    if child.returncode:
        print(json.dumps(status("failed", failed_stage=stage, returncode=child.returncode,
                                log=str(log_path))), flush=True)
        raise RuntimeError(f"{stage} failed; inspect {log_path}")
    print(json.dumps(status(stage + "_complete", log=str(log_path))), flush=True)


def main():
    wait_signals()
    if not (OUT / "COMPLETE.json").exists():
        execute("full_train", ["train", "--out", OUT, "--epochs", "24"])
    complete = read(OUT / "COMPLETE.json")
    assert complete["epochs"] == 24 and not complete["fixed_final"]
    selected = read(OUT / "SELECTED.json")
    epoch = selected["epoch"]
    assert 1 <= epoch <= 24
    save(OUT / "OOF_EPOCH_FREEZE.json", {"epochs": epoch, "selected_full_fit_epoch": epoch,
        "full_selection_sha256": sha(OUT / "SELECTED.json"),
        "rule": "same fixed final epoch for every fold; no held-out fit targets used for selection"})
    print(json.dumps(status("full_fit_selected", epoch=epoch,
                            validation=selected["evaluation"])), flush=True)
    folds = []
    for fold in range(3):
        out = RUN / f"whitebox_oof_fold_{fold}"
        fit_ids = RUN / f"fusion/folds/fold_{fold}_fit_ids.json"
        held_ids = RUN / f"fusion/folds/fold_{fold}_held_ids.json"
        if not (out / "COMPLETE.json").exists():
            execute(f"fold_{fold}_train", ["train", "--out", out, "--fit-ids", fit_ids,
                                            "--epochs", epoch, "--fixed-final"])
        fold_complete = read(out / "COMPLETE.json")
        assert fold_complete["epochs"] == epoch and fold_complete["fixed_final"]
        execute(f"fold_{fold}_predict_heldout", ["predict", "--out", out,
                                                "--predict-partition", "fit", "--predict-ids", held_ids])
        prediction = out / "token_predictions_fit.jsonl"
        receipt = read(prediction.with_suffix(".manifest.json"))
        assert receipt["all_predictions_out_of_fit"] and receipt["training_member_count"] == 0
        assert receipt["responses"] == len(read(held_ids))
        folds.append({"fold": fold, "training_ids": str(out / "FIT_IDS.json"),
                      "predictions": str(prediction), "manifest": str(prediction.with_suffix(".manifest.json")),
                      "prediction_manifest": str(prediction.with_suffix(".manifest.json")),
                      "run_manifest": str(out / "RUN.json"), "checkpoint": str(out / "best.pt")})
    save(OUT / "OOF_READY.json", {"epochs": epoch, "folds": folds,
                                  "validation": str(OUT / "best_validation_token_predictions.jsonl"),
                                  "test_predictions_created": False, "test_labels_read": False})
    print(json.dumps(status("complete", selected_epoch=epoch, fold_count=3,
                            oof_manifest=str(OUT / "OOF_READY.json"))), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            status("interrupted", error=repr(error))
        else:
            status("failed", error=repr(error))
        raise
