"""Independent window reconstruction and audit; no fitting or GPU calls.

Only training-source labels are read. Window boundaries depend on original item
and BPE boundaries, never on hallucination span boundaries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
import pickle
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R13 = ROOT.parent/"round13_generalization_diagnostics"
spec = importlib.util.spec_from_file_location("independent_audit13", R13/"results/audit13.py")
prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
a = prior.a; R10 = prior.R10
SIZES = (1, 4, 8)
METHODS = ("window_lr", "token_max")


def construct_windows(item, token_offsets, lexical_gold, k):
    """Pure reconstruction; lexical_gold maps original token index to 0/1."""
    inside = [i for i, (left, right) in enumerate(token_offsets) if right > item["start"] and left < item["end"]]
    if not inside: return []
    starts = range(len(inside)-k+1) if len(inside) >= k else [0]
    result = []
    for start in starts:
        indices = inside[start:start+k]
        scored = [j for j in indices if j in lexical_gold]
        if not scored: continue
        result.append({"k": k, "indices": indices, "eligible_indices": scored,
                       "start": max(item["start"], min(token_offsets[j][0] for j in indices)),
                       "end": min(item["end"], max(token_offsets[j][1] for j in indices)),
                       "gold": int(any(lexical_gold[j] for j in scored))})
    return result


def training_windows(items, tokens):
    token_by_row = defaultdict(dict)
    for t in tokens: token_by_row[t["row_id"]][t["token_index"]] = t
    windows = {k: [] for k in SIZES}; matrices = {k: [] for k in SIZES}
    for item in items:
        if not item["asserted_eligible"] or item["localization_status"] != "resolved": continue
        rid = item["row_id"]; g = a.read_json(R10/"data/generation_records"/(rid+".json"))
        with np.load(R10/"data/features"/(rid+".npz"), allow_pickle=False) as z: x = z["lookback_features"].copy()
        lookup = token_by_row[rid]; gold = {i: t["gold"] for i, t in lookup.items()}
        for index, (left, right) in enumerate(g["response_token_offsets"]):
            if right > item["start"] and left < item["end"]:
                a.require(not any(c.isalnum() for c in g["response"][left:right]) or index in lookup,
                          "Unknown lexical material cannot become negative window gold")
        for k in SIZES:
            for w in construct_windows(item, g["response_token_offsets"], gold, k):
                w.update({key:item[key] for key in ("row_id", "question_id", "group_id", "condition", "item_id")})
                w["category"] = item["category"]
                w["item_ids"] = [item["item_id"]]
                w["text"] = g["response"][w["start"]:w["end"]]
                windows[k].append(w); matrices[k].append(x[w["indices"]].mean(axis=0))
    matrices = {k:np.asarray(x, np.float32) for k,x in matrices.items()}
    a.require(len(windows[1]) == len(tokens), "K1 denominator does not reproduce original lexical tokens")
    for w, t, x in zip(windows[1], tokens, matrices[1]):
        a.require(w["row_id"] == t["row_id"] and w["indices"] == [t["token_index"]] and w["gold"] == t["gold"], "K1 token identity/gold changed")
    return windows, matrices


def binary_metrics(y, predictions):
    y = np.asarray(y, int); pred = np.asarray(predictions, bool)
    tp, fp = int(np.sum((y == 1)&pred)), int(np.sum((y == 0)&pred))
    fn, tn = int(np.sum((y == 1)&~pred)), int(np.sum((y == 0)&~pred))
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":tp/(tp+fp) if tp+fp else None,
            "recall":tp/(tp+fn) if tp+fn else None,"f1":2*tp/(2*tp+fp+fn) if tp+fn else None}


def merged_coverage(windows, decisions):
    """Union of original eligible BPE indices; overlap never increases coverage."""
    result = defaultdict(set)
    for w, positive in zip(windows, decisions):
        if positive: result[w["row_id"]].update(w["eligible_indices"])
    return result


def canonical(window):
    w = window
    return {"window_key": w["item_id"]+f'__w{w["k"]}__start{w["indices"][0]}',
            "width":w["k"], "actual_width":len(w["indices"]), "short_window":len(w["indices"])<w["k"],
            **{k:w[k] for k in ("row_id","item_ids","group_id","condition","category","start","end","text","gold")},
            "raw_token_indices":w["indices"],
            "token_keys":[w["row_id"]+f"__token{j}" for j in w["eligible_indices"]]}


def compare_metrics(rows, name, expected, scores=None, threshold=None):
    y = [r["gold"] for r in rows]
    if scores is None: scores = np.asarray([r["scores"][name] for r in rows])
    pred = [r["predictions"][name] for r in rows] if threshold is None else np.asarray(scores) >= threshold
    result = binary_metrics(y, pred)
    result.update(prior.ranking(y, scores))
    result.update(windows=len(rows), risk_windows=sum(y), alert_rate=float(np.mean(pred)), risk_rate=float(np.mean(y)))
    a.compare_values(result, expected, name)
    return result


def paired_intervals(rows):
    groups = sorted({r["group_id"] for r in rows})
    counts = np.zeros((len(groups),2,3), int)
    for j,g in enumerate(groups):
        rr = [r for r in rows if r["group_id"] == g]
        for k,name in enumerate(METHODS):
            c = binary_metrics([r["gold"] for r in rr], [r["predictions"][name] for r in rr])
            counts[j,k] = c["tp"],c["fp"],c["fn"]
    sample = np.random.default_rng(20260914).integers(0,len(groups),(2000,len(groups)))
    total = np.asarray([counts[ix].sum(0) for ix in sample]);tp,fp,fn = (total[:,:,j] for j in range(3))
    f1 = np.divide(2*tp,2*tp+fp+fn,out=np.full(tp.shape,np.nan,float),where=tp+fn>0)
    def ci(v):
        valid = v[np.isfinite(v)]
        return {"ci95":np.quantile(valid,[.025,.975]).tolist() if len(valid) else None,"defined_draws":len(valid)}
    return {"groups":len(groups),"methods":{name:ci(f1[:,j]) for j,name in enumerate(METHODS)},
            "window_lr_minus_token_max_f1":ci(f1[:,0]-f1[:,1])}


def merge_intervals(rows):
    result=[]
    for row in sorted(rows,key=lambda w:(w["start"],w["end"])):
        if result and row["start"] <= result[-1][1]: result[-1][1] = max(result[-1][1],row["end"])
        else: result.append([row["start"],row["end"]])
    return result


def audit_highlights(k, rows, items, tokens, expected, saved):
    usable = [i for i in items if i["asserted_eligible"] and i["localization_status"] == "resolved"]
    labels = {r["item_id"]:r for r in a.read_lines(R10/"data/annotations_train.jsonl")}
    by_token = {t["token_key"]:t for t in tokens}; by_item=defaultdict(list); rows_by_item=defaultdict(list)
    for t in tokens: by_item[t["item_ids"][0]].append(t)
    for r in rows: rows_by_item[r["item_ids"][0]].append(r)
    saved_lookup = {(r["width"],r["method"],r["item_id"]):r for r in saved}
    output={}
    for method in METHODS:
        covered_all=set();normal=risky=hit=full=total=0
        for item in usable:
            iid=item["item_id"];active=[r for r in rows_by_item[iid] if r["predictions"][method]]
            covered={key for r in active for key in r["token_keys"]};covered_all |= covered
            normal += int(item["gold"] == 0 and bool(active));risky += int(item["gold"] == 1 and bool(active))
            spans=labels[iid]["risk_spans"]
            for span in spans:
                required=set()
                for token in by_item[iid]:
                    if any(span["start"] <= token["start"]+j < span["end"] and c.isalnum() for j,c in enumerate(token["text"])):
                        required.add(token["token_key"])
                a.require(bool(required),"Risk region has no eligible token")
                total += 1;hit += bool(required&covered);full += required <= covered
            record=saved_lookup[k,method,iid]
            a.require(record["covered_token_keys"] == sorted(covered),"Highlight union mismatch")
            a.require(record["merged_intervals"] == merge_intervals(active),"Merged character region differs")
            a.require(record["text"] == item["text"] and record["item_start"] == item["start"] and record["item_gold"] == item["gold"],"Highlight item identity differs")
            a.require(record["gold_spans"] == [{key:s[key] for key in ("start","end","text")} for s in spans],"Highlight region gold differs")
        nnormal=sum(i["gold"]==0 for i in usable);nrisky=sum(i["gold"]==1 for i in usable)
        covered_risk=sum(by_token[key]["gold"] for key in covered_all)
        result={"eligible_answers":len(usable),"eligible_tokens":len(tokens),"highlighted_tokens":len(covered_all),
            "highlighted_token_fraction":len(covered_all)/len(tokens),"risk_tokens_covered":covered_risk,
            "risk_token_coverage":covered_risk/sum(t["gold"] for t in tokens),"normal_answer_alerts":normal,
            "normal_answers":nnormal,"normal_answer_false_alarm_rate":normal/nnormal,"risk_answer_alerts":risky,
            "risk_answers":nrisky,"risk_answer_any_alert_recall":risky/nrisky,"gold_regions":total,
            "gold_regions_hit_any_token":hit,"gold_region_any_overlap_recall":hit/total,"gold_regions_fully_covered":full}
        a.compare_values(result,expected[method],"coarse "+method);output[method]=result
    return output


def run(root):
    out=root/"results";done=a.read_json(out/"complete14.json")
    a.require(done["direct_fits"]==15 and done["reused_token_models"]==5 and not done["original_validation_or_test_labels_used"],"Scope differs")
    for name,checksum in done["files_sha256"].items():a.require(a.sha(out/name)==checksum,"Completed artifact changed: "+name)
    snap=a.read_json(out/"source_snapshot14.json");started=a.read_json(out/"started14.json")
    a.require(started["source_snapshot_sha256"]==a.sha(out/"source_snapshot14.json"),"Pre-run snapshot changed")
    a.require(snap["prior"]==a.read_json(R13/"results/source_snapshot13.json"),"Source graph changed")
    a.require(snap["prior_models_sha256"]==a.sha(R13/"results/fitted_folds.pkl") and snap["prior_completion_sha256"]==a.sha(R13/"results/complete13.json"),"R13 baseline changed")
    for base,files in ((root,snap["local_files_sha256"]),(R13,snap["prior"]["local_files_sha256"])):
        for name,checksum in files.items():a.require(a.sha(base/name)==checksum,"Frozen source changed: "+name)
    protocol=a.read_json(root/"protocol.json")
    a.require(protocol["widths"]==list(SIZES) and protocol["stride"]==1 and protocol["C"]==.01,"Window protocol changed")
    items,tokens,gold=prior.training_gold();windows,matrices=training_windows(items,tokens)
    old_models=pickle.loads((R13/"results/fitted_folds.pkl").read_bytes());models=pickle.loads((out/"fitted_folds.pkl").read_bytes())
    old_scores={r["token_key"]:r for r in a.read_lines(R13/"results/oof_scores.jsonl")}
    token_x,_=prior.previous.raw_matrix(tokens,"token");token_order={t["token_key"]:j for j,t in enumerate(tokens)}
    summary=a.read_json(out/"summary.json");a.require(summary["coverage"]==gold["coverage"],"Coverage changed")
    all_records=a.read_lines(out/"highlight_regions.jsonl")
    a.require(len(all_records)==len({(r['width'],r['method'],r['item_id']) for r in all_records})==3*2*195,"Highlight coverage differs")
    report={"schema":"round14-independent-audit-v1","status":"passed","gold":gold,"no_refitting_or_gpu":True,"widths":{},"models":{}}
    outputs={};catalogs={};by_key={}
    for k in SIZES:
        reconstructed=[canonical(w) for w in windows[k]];catalog=a.read_lines(out/f"window_catalog_k{k}.jsonl")
        a.require(reconstructed==catalog,"Original BPE catalog/labels differ: "+str(k));catalogs[k]=catalog
        scored=a.read_lines(out/f"window_scores_k{k}.jsonl");outputs[k]=scored;by_key[k]={r["window_key"]:r for r in scored}
        a.require(len(catalog)==len(scored)==len(by_key[k]),"Window OOF coverage/duplicates")
        for r in catalog:
            saved=by_key[k][r["window_key"]]
            a.require(all(saved[key]==v for key,v in r.items()),"Scored window identity differs")
            a.require(set(saved["scores"])==set(saved["predictions"])==set(METHODS),"Missing method")
            a.require(all(a.finite(s) for s in saved["scores"].values()),"Nonfinite window score")
    max_k1_error=0.;assigned={k:np.zeros(len(catalogs[k]),int) for k in SIZES}
    for fold,block in models.items():
        old=old_models[fold]
        for key in ("fit_groups","calibration_groups","evaluation_groups"):
            a.require(block[key]==old[key],"Changed R13 group split")
        a.require(not set(old['fit_groups'])&set(old['calibration_groups']) and not set(old['evaluation_groups'])&(set(old['fit_groups'])|set(old['calibration_groups'])),"Group leakage")
        ptoken=prior.probability(old["models"]["lb_c001"],token_x)
        target=sum(r["group_id"] in old["fit_groups"] for r in catalogs[1]);report["models"][str(fold)]={}
        a.require(set(block["models"])==set(SIZES),"Wrong window size model count")
        for k,model in block["models"].items():
            cc=catalogs[k];x=matrices[k]
            ix,ca,ev=[np.asarray([j for j,r in enumerate(cc) if r["group_id"] in old[name]],int) for name in ("fit_groups","calibration_groups","evaluation_groups")]
            for key,indices in (("fit_indices",ix),("calibration_indices",ca),("evaluation_indices",ev)):
                np.testing.assert_array_equal(model[key],indices)
            assigned[k][ev]+=1
            b,w,factors=prior.previous.direct_weights([cc[j] for j in ix],"token");w*=target/w.sum()
            np.testing.assert_allclose(model["base_weights"],b,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(model["loss_weights"],w,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(model["class_factors"],factors,atol=1e-12,rtol=1e-12)
            a.require(model["target_loss_mass"]==target and model["C"]==model["model"].C==.01 and model["components"] is None,"Window feature/model budget differs")
            prior.previous.check_scaler(model["scaler"],x[ix],b,f"fold{fold} k{k}")
            direct=prior.probability(model,x)
            pooled=np.asarray([max(ptoken[token_order[key]] for key in r["token_keys"]) for r in cc])
            score_arrays={"window_lr":direct,"token_max":pooled}
            errors={}
            for name,values in score_arrays.items():
                chosen=prior.independent_threshold([{"gold":cc[j]["gold"],"scores":{name:float(values[j])}} for j in ca],name)
                a.compare_values(chosen,model["thresholds"][name],"Calibration selection")
                detail=summary["folds"][str(fold)][str(k)][name]
                a.require(detail["threshold"]==model["thresholds"][name],"Threshold provenance differs")
                for stage,indices in (("fit",ix),("calibration",ca),("evaluation",ev)):
                    compare_metrics([cc[j] for j in indices],name,detail[stage],values[indices],chosen["threshold"])
                error=[]
                for j in ev:
                    saved=by_key[k][cc[j]["window_key"]]
                    a.require(saved["fold"]==fold,"Wrong outer fold")
                    diff=abs(saved["scores"][name]-values[j]);error.append(diff)
                    a.require(diff<=1e-9,"Frozen window probability not reproduced")
                    a.require(saved["predictions"][name]==(saved["scores"][name]>=chosen["threshold"]),"Wrong operating threshold")
                    if k==1:
                        original=old_scores[cc[j]["token_keys"][0]]
                        delta=abs(saved["scores"][name]-original["scores"]["lb_c001"]);max_k1_error=max(max_k1_error,delta)
                        a.require(delta<=1e-10 and saved["predictions"][name]==original["predictions"]["lb_c001"],"K1 did not reproduce R13")
                        a.require(chosen["threshold"]==old["models"]["lb_c001"]["threshold"],"K1 threshold changed")
                errors[name]=max(error)
            report["models"][str(fold)][str(k)]={"fit_windows":len(ix),"target_loss_mass":target,"groups_disjoint":True,
                "calibration_only_thresholds":True,"probability_max_error":errors,"preprocessing_and_weights":"passed"}
    a.require(len(models)==5 and all((seen==1).all() for seen in assigned.values()),"Windows not held out exactly once")
    for k in SIZES:
        rows=outputs[k];value=summary["widths"][str(k)];pos=sum(r["gold"] for r in rows)
        a.require(value["windows"]==len(rows) and value["positive_windows"]==pos,"Window denominator differs")
        a.require(value["short_windows"]==sum(r["short_window"] for r in rows),"Short window count differs")
        np.testing.assert_allclose([value["positive_window_rate"],value["always_alert_f1"]],[pos/len(rows),2*pos/(len(rows)+pos)],atol=1e-12,rtol=0)
        metrics={name:compare_metrics(rows,name,value["methods"][name]) for name in METHODS}
        boot=paired_intervals(rows);stated=value["paired_bootstrap"]
        a.require((stated["groups"],stated["draws"],stated["seed"])==(120,2000,20260914),"Bootstrap scope differs")
        for name,interval in boot["methods"].items():prior.previous.prior.check_interval(interval,stated["methods"][name],name)
        prior.previous.prior.check_interval(boot["window_lr_minus_token_max_f1"],stated["window_lr_minus_token_max_f1"],"paired difference")
        for name in METHODS:
            np.testing.assert_allclose(value["fold_mean_auroc"][name],np.mean([summary["folds"][str(f)][str(k)][name]["evaluation"]["auroc"] for f in range(5)]),atol=1e-12,rtol=0)
        coarse=audit_highlights(k,rows,items,tokens,value["coarse_highlight"],all_records)
        report["widths"][str(k)]={"windows":len(rows),"positive_windows":pos,"always_alert_f1":value["always_alert_f1"],
            "metrics":metrics,"paired_bootstrap":boot,"coarse_highlight":coarse}
    report["k1_replication"]={"rows":len(tokens),"max_score_difference_from_round13":max_k1_error,"thresholds_and_decisions_identical":True}
    report["summary_sha256"]=a.sha(out/"summary.json")
    a.write_json(out/"INDEPENDENT_AUDIT14.json",report)
    print(json.dumps({"status":"passed","direct_models":15,"k1_replication":report["k1_replication"],
        "widths":{k:{"f1":{n:m["f1"] for n,m in v["metrics"].items()},"paired_ci":v["paired_bootstrap"]["window_lr_minus_token_max_f1"]} for k,v in report["widths"].items()}},ensure_ascii=False))


def self_test():
    item = {"start": 0, "end": 6}
    # a , b . c ! : punctuation counts toward window length, not gold tokens.
    offsets = [(i,i+1) for i in range(6)]; labels = {0:0,2:1,4:0}
    one = construct_windows(item, offsets, labels, 1)
    a.require([w["indices"] for w in one] == [[0],[2],[4]], "K1 punctuation exclusion")
    four = construct_windows(item, offsets, labels, 4)
    a.require(len(four) == 3 and all(w["gold"] == 1 for w in four), "Fixed raw BPE window/gold overlap")
    eight = construct_windows(item, offsets, labels, 8)
    a.require(len(eight) == 1 and eight[0]["indices"] == list(range(6)), "Short item must form one short window")
    a.require(construct_windows(item, offsets, {}, 4) == [], "Pure punctuation window should be excluded")
    for w in four:w["row_id"]="r"
    a.require(merged_coverage(four,[True,True,True])["r"] == {0,2,4}, "Overlapping windows counted more than once")
    a.require(binary_metrics([1,0],[True,True])["f1"] == 2/3, "Window confusion calculation")
    print("SELF_TEST_PASSED: raw BPE windows, short answers, punctuation, overlap union")


if __name__ == "__main__":
    p = argparse.ArgumentParser();p.add_argument("stage",choices=["self-test","run"])
    p.add_argument("--root",type=Path,default=ROOT)
    args = p.parse_args()
    if args.stage == "self-test":self_test()
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):run(args.root.resolve())
