"""Development-only checks of within-message literal co-occurrence.

Read only the preceding frozen train/validation inputs. This does not fit a
model, inspect the old test, infer a causal process, or make an IID test claim.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
from scipy.special import expit


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(parent, out):
    plan, report = read(parent / "PLAN.json"), read(parent / "REPORT.json")
    assert sha(parent / "PLAN.json") == report["plan_sha256"]
    for name, expected in plan["inputs_sha256"].items():
        assert sha(parent / "inputs" / name) == expected
    name = "event_dag_0.001.json"
    assert sha(parent / "models" / name) == report["models_sha256"][name]
    saved = read(parent / "models" / name)
    theta = dict(zip(saved["names"], saved["theta"]))
    results = {}
    for split in ("train", "validation"):
        rows = read(parent / "inputs" / (split + ".json"))
        for stage in ("work", "publish"):
            records, templates = [], {}
            for row in rows:
                groups = row["prefixes"]["2"]["groups"]
                n = len(groups)
                c = np.asarray([g["coordinator_present"] for g in groups], float)
                w = np.asarray([g["worker_present"] for g in groups], float)
                y = w if stage == "work" else np.asarray(row["final_value_labels"], float)
                # Original public full-field projection verifies the raw vault.
                # Avoid assuming all vault values literally survive JSON here:
                # diagnostics condition only on the observed publish parent W.
                if stage == "work":
                    continue
                z = theta["event:publish:bias"] + theta["event:publish:parent_mean"] * w
                p = expit(z)
                count = int(y.sum())
                distribution = np.array([1.])
                for value in p:
                    distribution = np.convolve(distribution, [1-value, value])
                record = {"template": row["template_id"], "n": n,
                    "observed_any": int(count > 0), "expected_any": float(1-distribution[0]),
                    "observed_fields": count, "expected_fields": float(p.sum()),
                    "observed_pairs": count*(count-1)/2,
                    "expected_pairs": float((p.sum()**2-(p*p).sum())/2),
                    "observed_count": count, "expected_count_distribution": distribution.tolist()}
                records.append(record)
                templates.setdefault(row["template_id"], []).append(record)
            if not records:
                continue
            def summarize(items):
                observed, expected = np.zeros(17), np.zeros(17)
                for r in items:
                    observed[r["observed_count"]] += 1
                    expected[:len(r["expected_count_distribution"])] += r["expected_count_distribution"]
                return {"flows": len(items), "fields": sum(r["n"] for r in items),
                    **{key: float(np.mean([r[key] for r in items])) for key in (
                        "observed_any", "expected_any", "observed_fields", "expected_fields", "observed_pairs", "expected_pairs")},
                    "observed_count_histogram": observed.tolist(), "expected_count_histogram": expected.tolist()}
            results[split+"/"+stage] = {"overall": summarize(records), "templates": {k: summarize(v) for k,v in templates.items()}}
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / "AUDITOR.py")
    result = {"parent": str(parent), "parent_plan_sha256": sha(parent / "PLAN.json"),
        "model_sha256": sha(parent / "models" / name), "auditor_sha256": sha(out / "AUDITOR.py"),
        "scope": "Full observed Worker output conditions the final reply. Descriptive within-message checks, no causal or IID significance claim. Prior-used development only.",
        "new_api_calls": 0, "new_fit": False, "old_test_read": False, "results": results}
    (out / "REPORT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v["overall"] for k,v in results.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.parent.resolve(), args.out.resolve())
