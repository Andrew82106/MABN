"""Independent frozen QA-development audit: no fitting, project imports or GPU.

Only development fit/calibration exports and their frozen derived artifacts are
read. sklearn pickle objects become inert state holders. Matrix/scaler/score/PCA
checks use NumPy formulas, never fit/transform/predict methods or an SVD refit.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT / "data"
REPORT = OUT / "COEFFICIENT_AUDIT_QA.json"
METHODS = ("lookback_mean", "lookback_nll", "layerband_slots", "hidden64_lookback_nll")
CS = (0.001, 0.01, 0.1)
WIDTH = dict(zip(METHODS, (1024, 1025, 520, 1089)))
NFIT, NCAL, BATCH = 168123, 42241, 16384


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def write(report):
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


class Frozen:
    def __setstate__(self, state):
        self.__dict__.update(state)


class Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("sklearn."):
            return Frozen
        if module.startswith("numpy") or module in ("builtins", "collections"):
            return super().find_class(module, name)
        raise ValueError((module, name))


def unpickle(path):
    with Path(path).open("rb") as f:
        return Reader(f).load()


def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64)-np.asarray(b, np.float64))))


def near(a, b, tag, rtol=1e-10, atol=1e-11):
    assert np.allclose(a, b, rtol=rtol, atol=atol), (tag, maxerr(a, b))
    return maxerr(a, b)


def sigmoid(v):
    result = np.empty(v.shape, np.float64)
    positive = v >= 0
    result[positive] = 1/(1+np.exp(-v[positive]))
    e = np.exp(v[~positive])
    result[~positive] = e/(1+e)
    return result


def metadata():
    answers, windows = [], []
    for part in ("fit", "calibration"):
        with (DATA/f"answers_{part}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                a = json.loads(line)
                assert a["partition"] == part and a["eligible"] and a["quality"] == "good"
                assert a["label"] == int(bool(a["original_labels"]))
                answers.append({k:a[k] for k in ("response_id", "answer_id", "group_id", "partition", "token_count", "label")})
        with (DATA/f"windows_k4_{part}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                w = json.loads(line)
                assert w["partition"] == part and w["eligible"]
                windows.append({k:w[k] for k in ("response_id", "window_id", "answer_id", "group_id", "partition", "label", "token_indices")})
    assert len(answers) == 793 and len(windows) == NFIT+NCAL
    assert sum(a["partition"] == "fit" for a in answers) == 634
    assert all(w["partition"] == "fit" for w in windows[:NFIT])
    assert all(w["partition"] == "calibration" for w in windows[NFIT:])
    by_answer = {a["answer_id"]:a for a in answers}
    assert len(by_answer) == 793
    groups = {part:{a["group_id"] for a in answers if a["partition"] == part} for part in ("fit", "calibration")}
    assert len(groups["fit"]) == 615 and len(groups["calibration"]) == 154
    assert not groups["fit"] & groups["calibration"]
    indices = defaultdict(list)
    for j, w in enumerate(windows):
        a = by_answer[w["answer_id"]]
        assert all(w[k] == a[k] for k in ("response_id", "group_id", "partition"))
        ix = w["token_indices"]
        assert len(ix) == min(4,a["token_count"]) and ix == list(range(ix[0], ix[0]+len(ix)))
        assert 0 <= ix[0] and ix[-1] < a["token_count"]
        indices[w["answer_id"]].append(j)
    assert len(indices) == len(answers)
    saved = read(OUT/"score_index.json")
    for rows, source in ((saved["windows"],windows),(saved["answers"],answers)):
        assert len(rows) == len(source)
        assert all(all(r[k] == s[k] for k in r) for r,s in zip(rows,source))
    keys = read(OUT/"fit_keys.json")
    for field, key in (("window_ids","window_id"),("group_ids","group_id"),("answer_ids","answer_id")):
        assert keys[field] == [w[key] for w in windows[:NFIT]]
    assert len(set(keys["window_ids"])) == NFIT
    return answers, windows, indices


def audit_weights(windows):
    fit = windows[:NFIT]
    group_answers = defaultdict(set)
    answer_counts = Counter(w["answer_id"] for w in fit)
    grouped = defaultdict(list)
    for j,w in enumerate(fit):
        group_answers[w["group_id"]].add(w["answer_id"])
        grouped[w["group_id"]].append(j)
    n, ng = len(fit), len(group_answers)
    b = np.asarray([(n/ng)/(len(group_answers[w["group_id"]])*answer_counts[w["answer_id"]]) for w in fit])
    y = np.asarray([w["label"] for w in fit], int)
    mass = np.asarray([b[y == k].sum() for k in (0,1)])
    factors = b.sum()/(2*mass)
    u = b*factors[y]
    loss = np.empty(n)
    for ix in grouped.values():
        loss[ix] = (n/ng)*u[ix]/u[ix].sum()
    with np.load(OUT/"training_weights.npz", allow_pickle=False) as z:
        saved = {k:z[k].copy() for k in z.files}
    assert np.array_equal(y,saved["y"])
    answer_mass_error = max(abs(b[[j for j in grouped[g] if fit[j]["answer_id"] == a]].sum()-n/(ng*len(aa))) for g,aa in group_answers.items() for a in aa)
    result = {"fit_windows":n,"fit_answers":len(answer_counts),"fit_groups":ng,
              "fit_label_counts":np.bincount(y,minlength=2).tolist(),
              "base_max_abs":near(b,saved["base_weights"],"base"),
              "class_factors_max_abs":near(factors,saved["class_factors"],"class factors"),
              "loss_max_abs":near(loss,saved["loss_weights"],"loss"),
              "base_group_mass_max_abs":max(abs(b[ix].sum()-n/ng) for ix in grouped.values()),
              "base_answer_mass_max_abs":float(answer_mass_error),
              "loss_group_mass_max_abs":max(abs(loss[ix].sum()-n/ng) for ix in grouped.values()),
              "loss_total_mass":float(loss.sum()),"expected_loss_mass":NFIT,
              "formula":"b_i=(N/G)/(answers_in_group*eligible_windows_in_answer); f_y=sum(b)/(2*sum_class_y(b)); u=b*f_y; loss_i=(N/G)*u_i/sum_group(u)."}
    return saved["base_weights"], result


def design(arrays, method, left, right):
    if method == "lookback_mean":
        return np.asarray(arrays["base"][left:right,:1024])
    if method == "lookback_nll":
        return np.asarray(arrays["base"][left:right])
    if method == "layerband_slots":
        return np.asarray(arrays["slots"][left:right])
    return np.concatenate((arrays["base"][left:right], arrays["hidden"][left:right]), axis=1)


def moments(arrays, method, base):
    # Independent weighted Chan merge. sklearn 1.6.1 casts sample weights AND
    # the scalar previous count to input float32 at each partial_fit call.
    count = 0.0
    mean = np.zeros(WIDTH[method])
    var = np.zeros(WIDTH[method])
    for left in range(0,NFIT,BATCH):
        right = min(left+BATCH,NFIT)
        x = design(arrays,method,left,right).astype(np.float64)
        w = base[left:right].astype(np.float32).astype(np.float64)
        new_count = w.sum()
        center = w@x/new_count
        delta = x-center
        correction = w@delta
        batch_m2 = w@(delta*delta)-correction*correction/new_count
        count = float(np.float32(count))
        total = count+new_count
        merged = (count*mean+new_count*center)/total
        var = (count*var+batch_m2+(center-mean)**2*count*new_count/total)/total
        mean, count = merged, total
    eps = np.finfo(float).eps
    constant = var <= count*eps*var+(count*mean*eps)**2
    scale = np.sqrt(var)
    scale[constant] = 1
    return mean,var,scale,count


def audit_models(answers,windows,indices,arrays,base,report):
    weights_sha, keys_sha = sha(OUT/"training_weights.npz"), sha(OUT/"fit_keys.json")
    family_reports = {}
    for method in METHODS:
        objects = [unpickle(OUT/f"{method}_C{c:g}.pkl") for c in CS]
        mean,var,scale,count = moments(arrays,method,base)
        sc = objects[0]["scaler"]
        m = {"width":WIDTH[method],"fit_rows":NFIT,"calibration_rows":NCAL,
             "scaler_mean_max_abs":near(mean,sc.mean_,method+" mean",atol=2e-11),
             "scaler_variance_max_abs":near(var,sc.var_,method+" variance",atol=2e-10),
             "scaler_variance_max_relative":float(np.max(np.abs(var-sc.var_)/np.maximum(np.abs(sc.var_),1e-20))),
             "scaler_scale_max_abs":near(scale,sc.scale_,method+" scale",atol=2e-11),
             "scaler_sample_weight_count":float(sc.n_samples_seen_),
             "scaler_count_max_abs":near(count,sc.n_samples_seen_,method+" weighted count"),
             "float32_input_weight_mass_before_count_rounding":float(base.astype(np.float32).astype(float).sum()),
             "fit_standardized_exact":True,"candidates":[]}
        standard = np.load(OUT/"matrices"/(method+"_fit_standardized.npy"),mmap_mode="r",allow_pickle=False)
        assert standard.shape == (NFIT,WIDTH[method]) and standard.dtype == np.float32
        saved_scores = []
        for c,obj in zip(CS,objects):
            assert obj["C"] == c and obj["method"] == method and obj["width"] == WIDTH[method]
            assert obj["fit_only"] and (obj["fit_rows"],obj["fit_groups"],obj["fit_answers"]) == (NFIT,615,634)
            assert obj["weights_sha256"] == weights_sha and obj["fit_keys_sha256"] == keys_sha
            assert obj["pca_sha256"] == (sha(OUT/"hidden_pca.pkl") if method == "hidden64_lookback_nll" else None)
            for key in ("mean_","var_","scale_","n_samples_seen_"):
                assert np.array_equal(getattr(obj["scaler"],key),getattr(sc,key))
            model = obj["model"]
            assert np.array_equal(model.classes_,[0,1]) and model.coef_.shape == (1,WIDTH[method])
            assert model.solver == "liblinear" and model.penalty == "l2" and model.C == c
            assert model.class_weight is None and model.max_iter == 2000 and model.random_state == 20260924
            assert model.n_iter_.max() < 2000
            with np.load(OUT/f"{method}_C{c:g}_scores.npz",allow_pickle=False) as z:
                saved_scores.append((z["window_scores"].copy(),z["answer_scores"].copy()))
            m["candidates"].append({"C":c,"iterations":model.n_iter_.tolist(),"window_score_count":NFIT+NCAL,
                                     "window_max_abs":0.0,"answer_max_abs":0.0})
        replay = [np.empty(NFIT+NCAL) for _ in CS]
        for left in range(0,NFIT+NCAL,BATCH):
            right = min(left+BATCH,NFIT+NCAL)
            z = design(arrays,method,left,right).copy()
            assert z.dtype == np.float32 and np.isfinite(z).all()
            z -= sc.mean_
            z /= sc.scale_
            if left < NFIT:
                fit_right = min(right,NFIT)
                assert np.array_equal(z[:fit_right-left],standard[left:fit_right]),(method,"stored standardized rows")
            for j,obj in enumerate(objects):
                model = obj["model"]
                replay[j][left:right] = sigmoid((z@model.coef_.T+model.intercept_).ravel())
        for j, c in enumerate(CS):
            stored_window,stored_answer = saved_scores[j]
            assert stored_window.shape == (NFIT+NCAL,) and stored_answer.shape == (793,)
            err = near(replay[j],stored_window,(method,c,"manual scores"),rtol=2e-12,atol=2e-12)
            av = np.asarray([max(replay[j][indices[a["answer_id"]]]) for a in answers])
            ae = near(av,stored_answer,(method,c,"answer max"),rtol=2e-12,atol=2e-12)
            m["candidates"][j].update(window_max_abs=err,answer_max_abs=ae)
        family_reports[method] = m
        report["families"] = family_reports
        write(report)
        print("QA_COEFFICIENT_FAMILY_PASSED",method,flush=True)
    return family_reports


def audit_pca(answers,windows,indices,arrays,report):
    pca = unpickle(OUT/"hidden_pca.pkl")
    assert pca["fit_answers"] == 634 and pca["fit_groups"] == 615
    assert pca["seed"] == 20260924 and pca["n_iter"] == 3 and pca["whiten"] is False
    fit = [a for a in answers if a["partition"] == "fit"]
    counts = Counter(a["group_id"] for a in fit)
    identities, weights = [], []
    positions = {}
    for a in fit:
        n=a["token_count"];k=min(n,32)
        # Integer arithmetic expresses floor(linspace) independently.
        ix = [0] if k == 1 else [(j*(n-1))//(k-1) for j in range(k)]
        positions[a["response_id"]] = ix
        identities.extend({"response_id":a["response_id"],"group_id":a["group_id"],"token_index":j} for j in ix)
        weights.extend([1/(615*counts[a["group_id"]]*k)]*k)
    w = np.asarray(weights)
    assert identities == pca["sample"] and len(w) == pca["sample_count"] == 20288
    weight_error = near(w,pca["sample_weights"],"PCA weights",atol=1e-15)
    assert pca["mean"].shape == (4096,) and pca["components"].shape == (64,4096)
    orthogonal = near(pca["components"]@pca["components"].T,np.eye(64),"PCA orthogonal",atol=1e-11)
    manifest = read(DATA/"feature_manifest.json")
    records = manifest["records"]
    records = {r["response_id"]:r for r in records} if isinstance(records,list) else records
    assert set(records) == {a["response_id"] for a in answers}
    weighted_mean = np.zeros(4096)
    trace = 0.0
    projection_max = 0.0
    exact = True
    total_tokens = 0
    at = 0
    for number,a in enumerate(answers,1):
        rid=a["response_id"];rec=records[rid]
        assert rec["partition"] == a["partition"] and rec["response_tokens"] == a["token_count"]
        path=DATA/"features"/(rid+".npz")
        side=path.with_suffix(".json")
        assert sha(path) == rec["npz_sha256"] and sha(side) == rec["metadata_sha256"]
        meta=read(side)
        assert meta["signature_sha256"] == manifest["signature_sha256"] and meta["plan_sha256"] == rec["plan_sha256"]
        with np.load(path,allow_pickle=False) as z:
            hidden=z["hidden_last"]
        assert hidden.dtype == np.float32 and hidden.shape == (a["token_count"],4096)
        assert np.isfinite(hidden).all()
        if a["partition"] == "fit":
            xx=hidden[positions[rid]].astype(np.float64);k=len(xx)
            ww=w[at:at+k];at+=k
            weighted_mean += ww@xx
            trace += float(np.sum(ww[:,None]*(xx-pca["mean"])**2))
        projected=((hidden.astype(np.float64)-pca["mean"])@pca["components"].T).astype(np.float32)
        jj=indices[a["answer_id"]]
        mean_windows=np.asarray([projected[windows[j]["token_indices"]].mean(0) for j in jj],np.float32)
        saved=arrays["hidden"][jj]
        delta=near(mean_windows,saved,"all-token fixed PCA projection "+rid,rtol=1e-6,atol=2e-6)
        projection_max=max(projection_max,delta)
        exact=exact and np.array_equal(mean_windows,saved)
        total_tokens+=len(hidden)
        if number%100 == 0:
            print("QA_PCA_FIXED_FORWARD",number,793,flush=True)
    assert at == len(w)
    mean_error=near(weighted_mean,pca["mean"],"PCA fit-only weighted mean",atol=1e-10)
    explained=float(np.sum(pca["singular_values"]**2)/trace)
    explained_error=near(explained,pca["explained_variance_ratio_sum"],"PCA explained fraction",atol=1e-11)
    return {"fit_answers":634,"fit_groups":615,"sample_count":len(w),"sample_identity_exact":True,
            "sample_weight_max_abs":weight_error,"weighted_mean_max_abs":mean_error,
            "components_orthogonality_max_abs":orthogonal,"stored_singular_value_trace_ratio_max_abs":explained_error,
            "all_forward_answers":793,"all_forward_raw_tokens":total_tokens,"all_forward_windows":NFIT+NCAL,
            "projected_window_matrix_exact":exact,"projected_window_max_abs":projection_max,
            "refit_or_svd_called":False,"scope":"Verified deterministic fit-only sample/weights/mean and stored-components forward for every token/window; components were not re-estimated."}


def audit(report):
    complete=read(OUT/"complete.json")
    assert complete["status"] == "complete_development_only" and complete["official_test_opened"] is False
    assert complete["script_sha256"] == sha(ROOT/"src/run_development.py")
    for filename,h in complete["files_sha256"].items():
        assert sha(OUT/filename) == h,filename
    snapshot=read(OUT/"source_snapshot.json")
    for path,h in snapshot["files_sha256"].items():
        assert sha(path) == h,path
    mm=read(OUT/"matrix_manifest.json")
    assert mm["rows"] == NFIT+NCAL and mm["fit_rows"] == NFIT and mm["calibration_rows"] == NCAL
    assert mm["pca_sha256"] == sha(OUT/"hidden_pca.pkl")
    arrays={}
    for name,width in (("base",1025),("slots",520),("hidden",64)):
        path=OUT/"matrices"/(name+".npy")
        assert sha(path) == mm["files_sha256"][name]
        arrays[name]=np.load(path,mmap_mode="r",allow_pickle=False)
        assert arrays[name].shape == (NFIT+NCAL,width) and arrays[name].dtype == np.float32
    answers,windows,indices=metadata()
    base,report["weights"]=audit_weights(windows)
    write(report)
    print("QA_FIT_KEYS_AND_WEIGHTS_PASSED",flush=True)
    audit_models(answers,windows,indices,arrays,base,report)
    report["pca"]=audit_pca(answers,windows,indices,arrays,report)
    report["frozen_artifacts_verified"]=True
    report["total_lr_candidates"]=12
    report["total_lr_window_scores_replayed"]=12*(NFIT+NCAL)
    report["total_lr_answer_scores_replayed"]=12*793
    report["source_sha256"]={str(p.resolve()):sha(p) for p in (Path(__file__),ROOT/"src/run_development.py",ROOT/"development_protocol.json",OUT/"complete.json",OUT/"source_snapshot.json",OUT/"matrix_manifest.json",OUT/"training_weights.npz",OUT/"fit_keys.json",OUT/"hidden_pca.pkl",OUT/"score_index.json")}


if __name__ == "__main__":
    report={"status":"running","reviewer":"/root/data_build/extract_review","utc":datetime.now(timezone.utc).isoformat(),
            "no_new_fit":True,"no_gpu":True,"official_test_or_withheld_read":False,
            "method":"Frozen sklearn pickle objects replaced with inert state holders; independent NumPy weight/moment/projection/coef-sigmoid formulas. No project runner imports, fit, transform, predict or SVD calls.",
            "scaler_numeric_convention":"Installed sklearn1.6.1 partial_fit uses float32 input weights and recasts previous scalar effective count to float32 each16384-row block; audit reproduces this via independent Chan weighted-variance merging.",
            "limits":["Coefficient replay and frozen provenance do not independently rerun optimization or certify how arbitrary unrecorded executions were performed.","PCA fitted components are frozen inputs, not refit. Parent audits original Lookback formula/geometry separately.","Calibration is development selection performance, not a fresh final-test estimate."]}
    try:
        with threadpool_limits(limits=4):
            audit(report)
        report["status"]="passed"
    except Exception as e:
        report["status"]="failed"
        report["error"]=repr(e)
        report["traceback"]=traceback.format_exc()
        write(report)
        raise
    write(report)
    print("QA_COEFFICIENT_AUDIT_PASSED",sha(REPORT),flush=True)
