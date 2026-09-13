"""Complete pinned LUMINA formula on exact R16 actual-train answers.

No labels, fitted model, held-out content parsing, or new generation. The main
intervention jointly applies the two already frozen R19 body-token masks.
The default prepare stage is CPU-only; GPU stages require explicit invocation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
R19 = ROOT.parent / 'round19_three_signal_probe'
_spec = importlib.util.spec_from_file_location('r25_source19', R19/'src/extract19.py')
r19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r19)
base, lumina7 = r19.base, r19.lumina7
VERSION = 'r25-pinned-lumina-ipr-joint-body-v1'
EXPECTED = 602
CHUNK = lumina7.DEFAULT_CHUNK_SIZE
LAMBDA = 0.5
SCHEMA = {
    'token_ipr': 'float32[N], pinned official all-output-layer IPR',
    'token_mmd_single': 'float32[N,2], unchanged R19 single-body MMD in source order',
    'token_mmd_single_mean': 'float32[N], arithmetic mean of both frozen single-body MMD columns',
    'token_mmd_joint_body': 'float32[N], original versus simultaneous two-body intervention MMD',
    'token_lumina_single_mean': 'float32[N], .5*IPR-.5*single-source-mean MMD',
    'token_lumina_joint_body': 'float32[N], .5*IPR-.5*joint-body MMD; primary',
    'token_ids': 'int64[N], exact original response token IDs',
    'response_token_offsets': 'int32[N,2], original full-response character coordinates',
    'token_start': 'int32[N]',
    'token_end': 'int32[N]',
}


def protocol():
    return {
        'version': VERSION,
        'scope': 'Only R16 actual split=train: 301 questions, 278 groups, 602 fixed answers; exploratory existing event folds',
        'primary_method': 'lumina_joint_body_lambda05',
        'additional_method': 'lumina_single_mean_lambda05',
        'primary_lambda': LAMBDA,
        'secondary_calibration_lambda_grid': [.25, .5, .75],
        'secondary_selection': 'Separate named secondary only; calibration max min(window F1, answer F1), then window F1, window precision, prefer lambda=.5, then lower lambda; never replace primary',
        'formula': 'lambda*IPR-(1-lambda)*MMD; fixed primary lambda=.5; higher means higher risk ranking, not calibrated probability',
        'official_commit': lumina7.OFFICIAL_COMMIT,
        'official_code_sha256': lumina7.OFFICIAL_SHA256,
        'formula_version': lumina7.FORMULA_VERSION,
        'ipr': 'Unchanged lumina7.ipr_from_hidden over hidden_states[1:], all 28 output layers, including a second final norm on the final-normalized last state',
        'mmd': 'Unchanged top100 non-renormalized cosine-kernel squared MMD, including top-k mass-difference term',
        'joint_intervention': 'Apply the union of both disjoint R19 strictly interior body-token masks, each consuming its existing fixed donor stream; preserve titles, boundary tokens, question, template, prompt length and exact answer IDs/positions',
        'single_intervention_aggregation': 'Same unweighted arithmetic mean of both R19 source columns for every row; no source selection by condition, category, labels or output score',
        'donors': 'Existing frozen R19 reviewed R6 actual-train pool and source-slot streams; no new donor selection',
        'natural_random_context_or_full_title_replacement': False,
        'timing': 'Token j uses causal state P+j-1 under both equal-length prefixes, including first token at P-1',
        'dtype': 'Existing NF4/BF16 model path; hidden stays model dtype until pinned float32 probability/reduction routines',
        'chunk_size': CHUNK,
        'window_aggregation': 'Arithmetic mean of scores over original four raw BPE positions; stride1 and original short-window rule unchanged',
        'answer_aggregation': 'Maximum over all output-defined window scores for comparison to the current system; differs from original item/full-answer token-mean aggregation',
        'calibration': 'Original event evaluation fold f, calibration (f+1)%5, remaining3 fit; only calibration labels for risk-F1/precision/higher-threshold cutoffs; no classifier or PCA',
        'gold_and_eligibility': 'Unchanged R18/R22 geometry/assistant gold; safe refusals answer-negative and excluded from localization, unresolved excluded; extractor never reads any gold',
        'feature_rows': EXPECTED,
        'feature_schema': SCHEMA,
        'backbone_passes_per_row': 2,
        'selfcheck_additional_passes': 'Original repeat and two single-body checks on two fixed source-length-selected train rows only',
        'gpu_selfcheck': 'All raw original repeat statistics/IPR must match exactly; identity MMD <=1e-7; R19 single MMD max abs <=1e-6 with actual deltas recorded; fixed small tolerance cannot be relaxed after failure',
        'gpu_order': 'QA then R23 then R25, scheduled by parent; prepare and cpu-selfcheck never load model weights',
        'new_generation': False,
        'labels_used_in_extraction': False,
        'original_validation_test_used': False,
        'human_gold': False,
        'limits': ['Complete pinned formula adapted to NF4 Qwen and raw-BPE localization, not reproduction of all paper experiments',
                   'Joint-body token stream intervention preserves informative titles and is not a natural full-context retrieval',
                   'Repeatedly used development folds do not constitute a new final test'],
    }


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        assert base.read(path) == value, ('Frozen value changed', str(path))
    else:
        base.save(path, value)


def joint_prefix(original_prefix_ids, interventions):
    """Pure union of two verified, disjoint single-source interventions."""
    original = list(original_prefix_ids)
    assert original and len(interventions) == 2
    result = original.copy()
    occupied, changed_union = set(), set()
    for slot, entry in enumerate(interventions):
        assert entry['passage_index'] == slot
        mask = list(entry['eligible_body_token_positions'])
        assert mask and mask == sorted(set(mask))
        assert min(mask) >= 0 and max(mask) < len(original)
        assert not occupied.intersection(mask), 'Overlapping source masks'
        occupied.update(mask)
        replacement = list(entry['replacement_prefix_token_ids'])
        body = list(entry['replacement_body_token_ids'])
        assert len(replacement) == len(original) and len(body) == len(mask)
        actual_changes = {i for i,(a,b) in enumerate(zip(original,replacement)) if a != b}
        assert actual_changes == set(entry['changed_token_positions'])
        assert actual_changes and actual_changes <= set(mask), 'Single-source edit outside its mask'
        assert [replacement[i] for i in mask] == body
        changed_union.update(actual_changes)
        for i,t in zip(mask,body):
            result[i] = t
    changes = [i for i,(a,b) in enumerate(zip(original,result)) if a != b]
    assert set(changes) == changed_union
    assert all(original[i] == result[i] for i in range(len(original)) if i not in occupied)
    return result, sorted(occupied), changes


def mix_scores(ipr, mmd, lam=LAMBDA):
    ipr, mmd = np.asarray(ipr,np.float32), np.asarray(mmd,np.float32)
    assert ipr.shape == mmd.shape and 0 <= lam <= 1
    return np.float32(lam)*ipr - np.float32(1-lam)*mmd


def old_single(row, g, digest, old_plan, manifest):
    """Read and bind only this selected train row's old scalar MMD fields."""
    rid = row['row_id']; entry = manifest['records'][rid]
    p, side = R19/entry['npz'], R19/entry['json']
    assert base.sha(p) == entry['npz_sha256'] and base.sha(side) == entry['json_sha256']
    meta = base.read(side)
    assert entry['source_generation_sha256'] == meta['source_generation_sha256'] == digest
    assert meta['row_id'] == rid and meta['input_row_sha256'] == base.model7.digest(row)
    assert meta['actual_split'] == 'train' and meta['labels_or_detector_scores_read'] is False
    assert meta['response_regenerated'] is False
    assert meta['extraction_signature_sha256'] == manifest['extraction_signature_sha256']
    assert meta['arrays_sha256'] == entry['npz_sha256']
    assert meta['intervention_plan_sha256'] == base.model7.digest(old_plan)
    with np.load(p,allow_pickle=False) as z:
        assert z['token_ids'].tolist() == g['response_token_ids']
        assert z['response_token_offsets'].tolist() == g['response_token_offsets']
        values = z['token_lumina_mmd'].copy()
    assert values.shape == (len(g['response_token_ids']),2) and values.dtype == np.float32
    assert np.isfinite(values).all() and (values >= 0).all()
    return values, dict(entry)


def prepare():
    """CPU-only source verification and immutable joint-plan construction."""
    (ROOT/'data/features').mkdir(parents=True,exist_ok=True)
    immutable_json(ROOT/'protocol.json',protocol())
    rows, records, _ = r19.context()  # read-only, metadata split inspected before JSON content
    allowed = {r['row_id'] for r in rows}
    for pattern in ('*.json','*.npz'):
        for path in (ROOT/'data/features').glob(pattern):
            assert path.stem in allowed, ('Non-training output file',str(path))
    old_manifest = base.read(R19/'data/feature_manifest.json')
    assert old_manifest['complete'] and old_manifest['completed_count'] == EXPECTED
    assert set(old_manifest['records']) == allowed
    assert old_manifest['extraction_signature_sha256'] == base.model7.digest(old_manifest['extraction_signature'])
    assert r19.signature() == old_manifest['extraction_signature'], 'Old extractor/model/source signature changed'
    pm = base.read(R19/'data/perturbation_manifest.json')
    assert pm['plans_sha256'] == base.sha(R19/'data/perturbation_plans.jsonl')
    assert pm['donor_pool_sha256'] == base.sha(R19/'data/donor_pool.json')
    assert pm['extract19_sha256'] == base.sha(Path(r19.__file__))
    originals = [json.loads(s) for s in (R19/'data/perturbation_plans.jsonl').read_text('utf-8').splitlines() if s.strip()]
    assert [p['row_id'] for p in originals] == [r['row_id'] for r in rows]
    tok = r19.AutoTokenizer.from_pretrained(base.model7.MODEL,local_files_only=True)
    plans, singles, old_refs = {}, {}, {}
    for row,old in zip(rows,originals):
        rid = row['row_id']; g,digest = records[rid]
        assert old['source_generation_sha256'] == digest
        r19.validate_plan(row,g,old)
        layout = r19.layout(tok,row,g)
        for expected,intervention in zip(layout,old['interventions']):
            assert all(intervention[k] == v for k,v in expected.items())
            assert not set(intervention['replacement_body_token_ids']) & set(tok.all_special_ids)
        joint,union,changes = joint_prefix(g['input_token_ids'],old['interventions'])
        assert tok.decode(g['response_token_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False) == g['response']
        plan = {'row_id':rid,'actual_split':'train','source_generation_sha256':digest,
            'input_row_sha256':base.model7.digest(row),'original_intervention_plan_sha256':base.model7.digest(old),
            'original_prefix_token_ids':g['input_token_ids'],'response_token_ids':g['response_token_ids'],
            'response_token_offsets':g['response_token_offsets'],'joint_prefix_token_ids':joint,
            'joint_prefix_sha256':base.model7.digest(joint),'union_body_token_positions':union,
            'changed_token_positions':changes,'source_interventions':old['interventions'],
            'titles_template_question_answer_positions_unchanged':True,'natural_context_claim':False}
        plans[rid] = plan
        singles[rid], old_refs[rid] = old_single(row,g,digest,old,old_manifest)
    path = ROOT/'data/joint_plans.jsonl'
    text = ''.join(json.dumps(plans[row['row_id']],ensure_ascii=False)+'\n' for row in rows)
    if path.exists():
        assert path.read_text('utf-8') == text, 'Frozen joint plans changed'
    else:
        pending = path.with_suffix('.jsonl.pending'); pending.write_text(text,'utf-8'); pending.replace(path)
    signature = {
        'version':VERSION,'code_sha256':{str(p.resolve()):base.sha(p) for p in
            [Path(__file__),Path(r19.__file__),Path(lumina7.__file__),Path(base.__file__),Path(base.model7.__file__)]},
        'base_model_and_source_signature':base.signature(),
        'protocol_sha256':base.sha(ROOT/'protocol.json'),'joint_plans_sha256':base.sha(path),
        'r19_feature_manifest_sha256':base.sha(R19/'data/feature_manifest.json'),
        'r19_extraction_signature_sha256':old_manifest['extraction_signature_sha256'],
        'r19_plans_sha256':pm['plans_sha256'],'donor_pool_sha256':pm['donor_pool_sha256'],
        'donor_independence_review_sha256':base.sha(R19/'data/donor_independence_review.json'),
        'formula_version':lumina7.FORMULA_VERSION,'official_commit':lumina7.OFFICIAL_COMMIT,
        'official_code_sha256':lumina7.OFFICIAL_SHA256,'chunk_size':CHUNK,'lambda':LAMBDA,
        'selector':'R16 actual split=train; all outputs regardless of parse/refusal; no annotations opened',
        'expected_rows':EXPECTED,'schema':SCHEMA,
    }
    immutable_json(ROOT/'data/extraction_signature.json',signature)
    ordered = sorted(rows,key=lambda r:(len(records[r['row_id']][0]['input_token_ids'])+len(records[r['row_id']][0]['response_token_ids']),r['row_id']))
    selfcheck_ids = [ordered[0]['row_id'],ordered[-1]['row_id']]
    report = {'status':'prepared_cpu_only','rows':len(rows),'questions':301,'groups':278,
        'response_tokens':sum(len(g['response_token_ids']) for g,_ in records.values()),
        'old_single_mmd_cache_rows_validated':len(singles),'joint_masks_disjoint_and_union_only_all_rows':True,
        'raw_prefix_and_answer_exact_all_rows':True,'selfcheck_row_ids':selfcheck_ids,
        'selfcheck_selection':'Shortest and longest source+answer token length, ties row_id; source only, no labels',
        'source_generation_sha256':{rid:h for rid,(_,h) in records.items()},
        'old_r19_records':old_refs,'extraction_signature_sha256':base.model7.digest(signature),
        'labels_or_detector_scores_read':False,'original_validation_test_content_parsed':False,
        'model_weights_loaded':False,'gpu_used':False}
    immutable_json(ROOT/'data/preparation.json',report)
    print('R25_PREPARED',len(rows),'TOKENS',report['response_tokens'],'NO_GPU',flush=True)
    return tok,rows,records,plans,singles,old_refs,signature,selfcheck_ids


def validate(arrays,g,single):
    n=len(g['response_token_ids']); assert set(arrays)==set(SCHEMA)
    for key in SCHEMA:
        if key.startswith('token_') and key not in ('token_ids','token_start','token_end'):
            shape=(n,2) if key=='token_mmd_single' else (n,)
            assert arrays[key].shape==shape and arrays[key].dtype==np.float32
            assert np.isfinite(arrays[key]).all(),key
    assert np.array_equal(arrays['token_mmd_single'],single)
    assert np.array_equal(arrays['token_mmd_single_mean'],single.mean(axis=1))
    assert (arrays['token_ipr']>=0).all() and (arrays['token_mmd_joint_body']>=0).all()
    for variant in ('single_mean','joint_body'):
        assert np.array_equal(arrays['token_lumina_'+variant],mix_scores(arrays['token_ipr'],arrays['token_mmd_'+variant]))
    assert arrays['token_ids'].dtype==np.int64 and arrays['token_ids'].tolist()==g['response_token_ids']
    assert arrays['response_token_offsets'].dtype==np.int32 and arrays['response_token_offsets'].tolist()==g['response_token_offsets']
    for name,col in [('token_start',0),('token_end',1)]:
        assert arrays[name].dtype==np.int32 and np.array_equal(arrays[name],arrays['response_token_offsets'][:,col])


@torch.inference_mode()
def original_statistics(model,prefix,answer):
    final,layers=lumina7.collect_response_hidden(model,prefix,answer,capture_layers=True)
    stats=lumina7.final_statistics(model,final,answer,chunk_size=CHUNK)
    ipr=lumina7.ipr_from_hidden(model,layers,stats,chunk_size=CHUNK).numpy()
    del final,layers
    return stats,ipr


@torch.inference_mode()
def extract(model,g,plan,single,do_selfcheck=False):
    assert not model.training
    device=model.get_input_embeddings().weight.device
    assert device.type=='cuda', 'Only explicit GPU run stage may call production extract'
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    prefix,answer=g['input_token_ids'],g['response_token_ids']
    original,ipr=original_statistics(model,prefix,answer)
    altered=r19.stats(model,plan['joint_prefix_token_ids'],answer)
    joint=r19.mmd(model,original,altered)
    checks={}
    if do_selfcheck:
        repeated,repeated_ipr=original_statistics(model,prefix,answer)
        assert all(torch.equal(original[k],repeated[k]) for k in original)
        assert np.array_equal(ipr,repeated_ipr)
        identity=r19.mmd(model,original,repeated)
        assert float(np.abs(identity).max())<=1e-7
        diffs=[]
        for slot,intervention in enumerate(plan['source_interventions']):
            stats=r19.stats(model,intervention['replacement_prefix_token_ids'],answer)
            actual=r19.mmd(model,original,stats)
            delta=float(np.abs(actual-single[:,slot]).max())
            assert delta<=1e-6,('Old R19 single MMD mismatch; do not relax silently',slot,delta)
            diffs.append(delta)
        checks={'original_repeat_exact':True,'ipr_repeat_exact':True,'identity_mmd_max_abs':float(np.abs(identity).max()),
            'r19_single_mmd_max_abs_per_source':diffs,'first_token_causal_position':len(prefix)-1,
            'all_response_tokens':len(answer),'joint_mask_union_only':True}
    offsets=np.asarray(g['response_token_offsets'],np.int32)
    means=single.mean(axis=1)
    arrays={'token_ipr':ipr,'token_mmd_single':single.copy(),'token_mmd_single_mean':means,
        'token_mmd_joint_body':joint,'token_lumina_single_mean':mix_scores(ipr,means),
        'token_lumina_joint_body':mix_scores(ipr,joint),'token_ids':np.asarray(answer,np.int64),
        'response_token_offsets':offsets,'token_start':offsets[:,0].copy(),'token_end':offsets[:,1].copy()}
    validate(arrays,g,single);torch.cuda.synchronize()
    meta={'version':VERSION,'input_tokens':len(prefix),'response_tokens':len(answer),
        'seconds':time.perf_counter()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'baseline_backbone_passes':2,'selfcheck_extra_backbone_passes':3 if do_selfcheck else 0,
        'selfcheck':checks,'labels_or_detector_scores_read':False,'response_regenerated':False,
        'all_output_tokens_kept':True,'primary_lambda':LAMBDA,'formula_version':lumina7.FORMULA_VERSION}
    return arrays,meta


def expected(row,digest,plan,old_ref,sig):
    return {'row_id':row['row_id'],'actual_split':'train','source_generation_sha256':digest,
        'input_row_sha256':base.model7.digest(row),'joint_plan_sha256':base.model7.digest(plan),
        'r19_single_mmd_npz_sha256':old_ref['npz_sha256'],'r19_single_mmd_json_sha256':old_ref['json_sha256'],
        'extraction_signature_sha256':base.model7.digest(sig)}


def cached(row,g,digest,plan,single,old_ref,sig):
    path=ROOT/'data/features'/(row['row_id']+'.npz');side=path.with_suffix('.json')
    if not side.exists():return None
    meta=base.read(side)
    for k,v in expected(row,digest,plan,old_ref,sig).items():assert meta[k]==v,(row['row_id'],k)
    assert base.sha(path)==meta['arrays_sha256']
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    validate(arrays,g,single)
    return arrays,meta


def save_row(row,g,digest,plan,single,old_ref,sig,arrays,meta):
    assert base.read(ROOT/'data/extraction_signature.json')==sig
    for filename,h in sig['code_sha256'].items():assert base.sha(filename)==h
    assert base.sha(base.SOURCE/'data/generation_records'/(row['row_id']+'.json'))==digest
    validate(arrays,g,single)
    path=ROOT/'data/features'/(row['row_id']+'.npz')
    base.save_arrays(path,arrays)
    meta.update(expected(row,digest,plan,old_ref,sig),arrays_sha256=base.sha(path))
    base.save(path.with_suffix('.json'),meta)


def manifest(rows,records,plans,singles,old_refs,sig):
    result={};missing=[]
    for row in rows:
        rid=row['row_id'];g,digest=records[rid]
        value=cached(row,g,digest,plans[rid],singles[rid],old_refs[rid],sig)
        if value is None:missing.append(rid);continue
        path=ROOT/'data/features'/(rid+'.npz');side=path.with_suffix('.json')
        result[rid]={'npz':str(path.relative_to(ROOT)),'json':str(side.relative_to(ROOT)),
            'npz_sha256':base.sha(path),'json_sha256':base.sha(side),'source_generation_sha256':digest,
            'response_tokens':len(g['response_token_ids']),'split':'train'}
    value={'version':VERSION,'complete':len(result)==EXPECTED,'completed_count':len(result),'generated':len(result),
        'expected_count':EXPECTED,'records':result,'missing_rows':missing,'schema':SCHEMA,
        'total_response_tokens_completed':sum(r['response_tokens'] for r in result.values()),
        'extraction_signature':sig,'extraction_signature_sha256':base.model7.digest(sig),
        'labels_or_detector_scores_read':False,'original_validation_test_used':False,
        'response_regenerated':False,'primary_lambda':LAMBDA}
    path=ROOT/'data/feature_manifest.json'
    if not path.exists() or base.read(path)!=value:base.save(path,value)
    return value


def run(selfcheck_only=False):
    tok,rows,records,plans,singles,old_refs,sig,selfcheck_ids=prepare()
    report=base.read(ROOT/'data/cpu_selfcheck.json')
    assert report['status']=='passed', 'CPU formula and intervention checks must pass first'
    assert report['extract25_sha256']==base.sha(Path(__file__)), 'CPU check predates current extractor code'
    assert report['test_script_sha256']==base.sha(ROOT/'tests/test_formula25.py')
    fm=manifest(rows,records,plans,singles,old_refs,sig)
    if fm['complete']:
        print('R25_ALREADY_COMPLETE_VERIFIED',flush=True);return
    _,model=base.model7.load_model()
    byid={r['row_id']:r for r in rows};checks=[]
    for rid in selfcheck_ids:
        row=byid[rid];g,digest=records[rid]
        value=cached(row,g,digest,plans[rid],singles[rid],old_refs[rid],sig)
        if value is not None:
            arrays,meta=value
            assert meta['selfcheck'].get('ipr_repeat_exact') is True, 'Engineering row cache lacks selfcheck'
        else:
            arrays,meta=extract(model,g,plans[rid],singles[rid],do_selfcheck=True)
            save_row(row,g,digest,plans[rid],singles[rid],old_refs[rid],sig,arrays,meta)
        checks.append({'row_id':rid,'selfcheck':meta['selfcheck'],'seconds':meta['seconds'],
            'response_tokens':meta['response_tokens'],'peak_allocated_gib':meta['peak_allocated_gib']})
    base.save(ROOT/'data/gpu_selfcheck.json',{'status':'passed','records':checks,
        'extraction_signature_sha256':base.model7.digest(sig),'calibration_or_gold_used':False})
    print('R25_GPU_SELFCHECK_PASSED',flush=True)
    if selfcheck_only:
        manifest(rows,records,plans,singles,old_refs,sig);return
    started=time.perf_counter()
    for index,row in enumerate(rows):
        rid=row['row_id'];g,digest=records[rid]
        if cached(row,g,digest,plans[rid],singles[rid],old_refs[rid],sig) is not None:continue
        arrays,meta=extract(model,g,plans[rid],singles[rid])
        save_row(row,g,digest,plans[rid],singles[rid],old_refs[rid],sig,arrays,meta)
        if (index+1)%50==0:
            fm=manifest(rows,records,plans,singles,old_refs,sig)
            print('R25_EXTRACT',fm['completed_count'],EXPECTED,'SECONDS',round(time.perf_counter()-started,1),flush=True)
    fm=manifest(rows,records,plans,singles,old_refs,sig);assert fm['complete']
    base.save(ROOT/'data/completion.json',{'status':'complete','seconds_loop':time.perf_counter()-started,
        'manifest_sha256':base.sha(ROOT/'data/feature_manifest.json'),
        'extraction_signature_sha256':base.model7.digest(sig),'labels_or_detector_scores_read':False})
    print('R25_COMPLETE',EXPECTED,'TOKENS',fm['total_response_tokens_completed'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','selfcheck','run'])
    args=parser.parse_args()
    if args.stage=='prepare':
        _,rows,records,plans,singles,old_refs,sig,_=prepare()
        manifest(rows,records,plans,singles,old_refs,sig)
    else:run(args.stage=='selfcheck')
