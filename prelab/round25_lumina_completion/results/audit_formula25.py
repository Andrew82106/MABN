"""Independent saved-array/source/geometry audit; no model imports or fitting."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'; PRE=ROOT.parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def rows(p):return [json.loads(x) for x in Path(p).read_text('utf-8').splitlines() if x]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    t=time.perf_counter();fm=read(ROOT/'data/feature_manifest.json'); sig=read(ROOT/'data/extraction_signature.json')
    assert fm['complete'] and fm['completed_count']==fm['expected_count']==602
    assert fm['extraction_signature']==sig and fm['extraction_signature_sha256']==digest(sig)
    for p,h in sig['code_sha256'].items():assert sha(p)==h,p
    assert sha(ROOT/'protocol.json')==sig['protocol_sha256']
    assert sha(ROOT/'data/joint_plans.jsonl')==sig['joint_plans_sha256']
    plans={p['row_id']:p for p in rows(ROOT/'data/joint_plans.jsonl')}
    r19=PRE/'round19_three_signal_probe';old=read(r19/'data/feature_manifest.json')
    assert sha(r19/'data/feature_manifest.json')==sig['r19_feature_manifest_sha256']
    assert set(plans)==set(fm['records'])==set(old['records'])
    windows=rows(OUT/'candidate_windows25.jsonl');items=rows(OUT/'answer_index25.jsonl')
    assert windows==rows(PRE/'round22_evidence_verification/results/candidate_windows22.jsonl')
    assert items==rows(PRE/'round22_evidence_verification/results/answer_index22.jsonl')
    assert len(windows)==12222 and len(items)==602
    assert {a['row_id'] for a in items}==set(plans)
    byrow={}; nt=0; changes=0; token_ranges=defaultdict(list); hashes={}
    for rid,entry in fm['records'].items():
        p=plans[rid];side=ROOT/entry['json'];arr=ROOT/entry['npz'];s=read(side)
        assert sha(side)==entry['json_sha256'] and sha(arr)==entry['npz_sha256']==s['arrays_sha256']
        assert p['actual_split']==entry['split']==s['actual_split']=='train'
        assert s['row_id']==rid and s['joint_plan_sha256']==digest(p)
        assert s['input_row_sha256']==p['input_row_sha256']
        assert s['extraction_signature_sha256']==fm['extraction_signature_sha256']
        gp=PRE/'round16_dataset_expansion/data/generation_records'/(rid+'.json');g=read(gp)
        assert sha(gp)==entry['source_generation_sha256']==s['source_generation_sha256']==p['source_generation_sha256']
        assert g['response_token_ids']==p['response_token_ids'] and g['response_token_offsets']==p['response_token_offsets']
        assert g['input_token_ids']==p['original_prefix_token_ids']
        prefix=p['original_prefix_token_ids'];union=set();joint=prefix.copy();changed=set()
        for slot,e in enumerate(p['source_interventions']):
            mask=e['eligible_body_token_positions'];alt=e['replacement_prefix_token_ids'];body=e['replacement_body_token_ids']
            assert e['passage_index']==slot and len(mask)==len(body) and len(alt)==len(prefix)
            assert mask==sorted(set(mask)) and not union.intersection(mask)
            delta={j for j,(a,b) in enumerate(zip(prefix,alt)) if a!=b}
            assert delta==set(e['changed_token_positions']) and delta<=set(mask)
            assert [alt[j] for j in mask]==body
            for j,v in zip(mask,body):joint[j]=v
            union.update(mask);changed.update(delta)
        assert joint==p['joint_prefix_token_ids'] and sorted(union)==p['union_body_token_positions']
        assert sorted(changed)==p['changed_token_positions']
        assert all(joint[j]==prefix[j] for j in range(len(prefix)) if j not in union)
        assert digest(joint)==p['joint_prefix_sha256'];changes+=len(changed)
        oldentry=old['records'][rid];op=r19/oldentry['npz'];os=r19/oldentry['json']
        assert sha(op)==oldentry['npz_sha256']==s['r19_single_mmd_npz_sha256']
        assert sha(os)==oldentry['json_sha256']==s['r19_single_mmd_json_sha256']
        with np.load(op,allow_pickle=False) as z:previous=z['token_lumina_mmd'].copy()
        with np.load(arr,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        n=len(g['response_token_ids']);nt+=n
        assert a['token_ids'].dtype==np.int64 and a['token_ids'].tolist()==g['response_token_ids']
        assert a['response_token_offsets'].dtype==np.int32 and a['response_token_offsets'].tolist()==g['response_token_offsets']
        assert np.array_equal(a['token_start'],a['response_token_offsets'][:,0])
        assert np.array_equal(a['token_end'],a['response_token_offsets'][:,1])
        for k in ('token_ipr','token_mmd_single_mean','token_mmd_joint_body','token_lumina_single_mean','token_lumina_joint_body'):
            assert a[k].shape==(n,) and a[k].dtype==np.float32 and np.isfinite(a[k]).all()
        assert a['token_mmd_single'].shape==(n,2) and np.array_equal(a['token_mmd_single'],previous)
        assert np.array_equal(a['token_mmd_single_mean'],previous.mean(axis=1))
        assert (a['token_ipr']>=0).all() and (a['token_mmd_joint_body']>=0).all()
        scores={'single_lambda05':np.float32(.5)*a['token_ipr']-np.float32(.5)*previous.mean(axis=1)}
        for lam,k in [(.25,'joint_lambda025'),(.5,'joint_lambda05'),(.75,'joint_lambda075')]:
            scores[k]=np.float32(lam)*a['token_ipr']-np.float32(1-lam)*a['token_mmd_joint_body']
        assert np.array_equal(scores['single_lambda05'],a['token_lumina_single_mean'])
        assert np.array_equal(scores['joint_lambda05'],a['token_lumina_joint_body'])
        byrow[rid]=scores
        for k in ('token_ipr','token_mmd_single_mean','token_mmd_joint_body'):token_ranges[k].extend(a[k].tolist())
        assert not s['labels_or_detector_scores_read'] and not s['response_regenerated'] and s['all_output_tokens_kept']
        for path in (arr,side):hashes[str(path.resolve())]=sha(path)
    assert nt==14968==fm['total_response_tokens_completed']
    with np.load(OUT/'formula_window_scores25.npz',allow_pickle=False) as z:stored={k:z[k].copy() for k in z.files}
    computed={k:[] for k in stored};byitem=defaultdict(list)
    for i,w in enumerate(windows):
        assert w['split']=='train';ix=np.asarray(w['raw_token_indices'],int)
        assert len(ix)==w['actual_width'] and len(ix)>0 and np.all(np.diff(ix)==1)
        assert len(ix)==4 or w['short_window']
        for k in computed:computed[k].append(float(byrow[w['row_id']][k][ix].mean()))
        byitem[w['item_ids'][0]].append(i)
    for k in stored:assert np.array_equal(stored[k],np.asarray(computed[k],np.float64)),k
    oofw=rows(OUT/'window_scores_oof25.jsonl');oofa=rows(OUT/'answer_scores_oof25.jsonl')
    wi={w['window_key']:w for w in oofw};ai={a['item_id']:a for a in oofa}
    assert len(wi)==len(windows) and len(ai)==len(items)
    methods=list(oofw[0]['scores']); safe_checks=0
    for f in range(5):
        cal=read(OUT/f'fold_{f}_calibration25.json')
        with np.load(OUT/f'fold_{f}_scores25.npz',allow_pickle=False) as z:full={k:z[k].copy() for k in z.files}
        lam=cal['secondary_selected_lambda'];key={.25:'joint_lambda025',.5:'joint_lambda05',.75:'joint_lambda075'}[lam]
        assert np.array_equal(full['lumina_single_mean_lambda05'],stored['single_lambda05'])
        assert np.array_equal(full['lumina_joint_body_lambda05'],stored['joint_lambda05'])
        assert np.array_equal(full['lumina_joint_body_lambda_cal'],stored[key])
        eg=set(cal['groups']['evaluation_groups'])
        for i,w in enumerate(windows):
            if w['group_id'] not in eg:continue
            s=wi[w['window_key']];assert s['fold']==f
            for k in methods:
                assert s['scores'][k]==float(full[k][i])
                assert s['predictions'][k]==bool(full[k][i]>=cal['thresholds'][k]['window']['threshold'])
        for a in items:
            if a['group_id'] not in eg:continue
            s=ai[a['item_id']];assert s['fold']==f;ix=byitem[a['item_id']]
            for k in methods:
                v=float(np.max(full[k][ix])) if ix else None
                assert s['scores'][k]==v
                assert s['predictions'][k]==(bool(v>=cal['thresholds'][k]['answer']['threshold']) if v is not None else None)
                if a['reviewed_safe_refusal']:
                    assert a['main_eligible'] and a['gold']==0 and ix and all(not windows[j]['main_eligible'] for j in ix)
                    safe_checks+=1
    assert safe_checks==1170
    source=read(OUT/'source_snapshot25.json')
    for f,h in source['files_sha256'].items():assert sha(f)==h,f
    for name in ('calibration_freeze25.json','complete25.json'):
        for f,h in read(OUT/name)['files_sha256'].items():assert sha(OUT/f)==h,f
    cpu=read(ROOT/'data/cpu_selfcheck.json');gpu=read(ROOT/'data/gpu_selfcheck.json')
    assert cpu['status']==gpu['status']=='passed'
    assert gpu['extraction_signature_sha256']==fm['extraction_signature_sha256']
    assert len(gpu['records'])==2
    for r in gpu['records']:
        c=r['selfcheck'];assert c['original_repeat_exact'] and c['ipr_repeat_exact']
        assert c['identity_mmd_max_abs']==0 and c['r19_single_mmd_max_abs_per_source']==[0.,0.]
    report={'status':'passed','blockers':[],'scope':'All602 R16 actual-train saved token arrays and source hashes; no original validation/test or human-QA test parsed.',
      'feature_rows':602,'response_tokens':nt,'formula_token_score_comparisons':nt*2,'single_source_MMD_values_exact_R19':nt*2,
      'all_joint_plans_exact_union_of_original_two_body_masks':True,'joint_changed_positions_total':changes,
      'frozen_source_answer_ids_offsets_and_feature_arrays_bound':True,'raw_four_BPE_window_means_exact':len(windows)*4,
      'all10_methods_all_candidate_answermax_and_OOF_exact':True,'safe_refusal_max_checks':safe_checks,
      'main_answer_count':sum(a['main_eligible'] for a in items),'main_window_count':sum(w['main_eligible'] for w in windows),
      'GPU_selfcheck_records_verified':2,'CPU_formula_selfcheck_record_verified':True,
      'token_feature_ranges':{k:{'min':min(v),'max':max(v),'mean':float(np.mean(v))} for k,v in token_ranges.items()},
      'implementation_review':{'IPR':'Original lumina7.ipr_from_hidden over all output layers; saved final statistics; extra final norm behavior retained.',
       'MMD':'Original top100 unrenormalized cosine kernel MMD, not IPR substituted; same response states P+j-1 on original/joint prefixes.',
       'combined_score':'float32 lambda*IPR-(1-lambda)*MMD; higher ranks as more risk, signed and not probability.',
       'intervention':'Only the union of both original R19 body-token masks changes. Titles/question/template and token counts remain unchanged; not natural full-context replacement.'},
      'limitations':['Saved IPR/MMD raw features checked for identity and formula integrity; no new full-model replay of all602 raw features in this CPU audit.',
       'Pinned CPU oracle and two production GPU checks are verified saved records, not independently reexecuted GPU checks.',
       'Original LUMINA answer aggregation differs: current experiment mean over raw4BPE then max over all windows.',
       'Repeatedly inspected assistant-labeled development OOF, not a fresh final test or a human-QA baseline replication.'],
      'no_fit':True,'no_GPU':True,'seconds':time.perf_counter()-t,
      'files_sha256':{str(p.resolve()):sha(p) for p in [Path(__file__),ROOT/'src/extract25.py',ROOT/'src/run25.py',ROOT/'data/feature_manifest.json',ROOT/'data/extraction_signature.json',ROOT/'data/joint_plans.jsonl',ROOT/'data/cpu_selfcheck.json',ROOT/'data/gpu_selfcheck.json',OUT/'formula_window_scores25.npz',OUT/'source_snapshot25.json',OUT/'complete25.json']}}
    (OUT/'FORMULA_AUDIT25.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','feature_rows','response_tokens','raw_four_BPE_window_means_exact','safe_refusal_max_checks','seconds')}))
if __name__=='__main__':main()
