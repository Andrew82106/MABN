"""ReDeEP/LUMINA on the fixed same-domain rows; never opens test labels.

prepare and cpu-check are CPU-only. extract is an explicit GPU command.
The original prefix, answer tokens and offsets are retained exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
BENCH = RUN.parents[1]
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(BENCH / 'src'))
import whitebox_signals as common
import redeep_formal_core_v1 as formula
import run_redeep_formal_baseline_v1 as redeep
import lumina_qa_signals as lumina

PARTITIONS = ('fit', 'validation', 'test')
COUNTS = {'fit': 1292, 'validation': 271, 'test': 279}
SEED = 20260928
COPY_HEADS = BENCH / 'third_party/ReDEeP-ICLR/ReDeEP/log/test_llama2_7B/topk_heads.json'
EXPECTED_HEAD_SHA = '633a37382dfe46242028cea3d4f28c6bb2102da1859572f6077a39a612228a4b'
SOURCES = [Path(__file__), Path(redeep.__file__), Path(formula.__file__),
           Path(lumina.__file__), lumina.PORT_PATH, Path(common.__file__)]
sha, digest, readl, publish = common.sha, common.digest, common.readl, common.publish


def write_jsonl(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.jsonl.pending')
    with pending.open('w', encoding='utf-8', newline='\n') as out:
        for row in values:
            out.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
    pending.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def bound_files():
    return {str(RUN / 'data' / name): sha(RUN / 'data' / name)
            for name in ['MANIFEST.json', 'split_assignments.jsonl'] +
            [f'{p}/{name}.jsonl' for p in PARTITIONS for name in ('inputs', 'geometry')]}


def check_geometry(tok, row, frozen):
    g = common.native.geometry(tok, row)
    for left, right in [('input_ids','full_input_ids'),('answer_token_ids','response_token_ids'),
                        ('answer_token_positions','answer_token_positions'),
                        ('response_token_offsets','response_token_offsets'),
                        ('response_token_offsets_raw','response_token_offsets_raw'),
                        ('context_token_positions','context_token_positions')]:
        assert g[left] == frozen[right], (row['response_id'], left)
    first = g['answer_token_positions'][0]
    assert g['answer_token_positions'] == list(range(first, len(g['input_ids'])))
    assert g['answer_token_ids'] == g['input_ids'][first:]
    assert g['input_sha256'] == frozen['input_sha256'] == digest(row)
    assert len(g['context_token_positions']) >= 10
    assert all(i < first for i in g['context_token_positions'])
    assert g['response_token_offsets'][-1][1] == len(row['original_response'])
    return g


def prepare():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(redeep.MODEL_PATH, use_fast=True, local_files_only=True)
    inputs = {p: readl(RUN / f'data/{p}/inputs.jsonl') for p in PARTITIONS}
    geometries = {p: {r['response_id']: r for r in readl(RUN / f'data/{p}/geometry.jsonl')} for p in PARTITIONS}
    members = {r['response_id']: r for r in readl(RUN / 'data/split_assignments.jsonl')}
    # Deduplicate donor material; selection uses no answer, label or score.
    donors = {}
    for row in inputs['fit']:
        key = digest(row['retrieved_passages'])
        donors.setdefault(key, {'material_hash':key, 'material':row['retrieved_passages'],
            'response_id':row['response_id'], 'group_id':members[row['response_id']]['group_id'],
            'titles':sorted({d['title'].strip().casefold() for d in row['documents']}),
            'texts':sorted({digest(d['text']) for d in row['documents']})})
    donor_list = sorted(donors.values(), key=lambda x:x['material_hash'])
    plans, summaries = [], {}
    limit = int(load_json(redeep.MODEL_PATH / 'config.json')['max_position_embeddings'])
    for partition in PARTITIONS:
        assert len(inputs[partition]) == len(geometries[partition]) == COUNTS[partition]
        maxima = {'original':0, 'random':0, 'answer_tokens':0}
        for row in inputs[partition]:
            rid = row['response_id']; g = check_geometry(tok, row, geometries[partition][rid])
            first = g['answer_token_positions'][0]
            plan = {'response_id':rid, 'partition':partition, 'group_id':members[rid]['group_id'],
                'input_sha256':digest(row), 'answer_sha256':hashlib.sha256(row['original_response'].encode()).hexdigest(),
                'answer_token_ids':g['answer_token_ids'], 'response_token_offsets':g['response_token_offsets'],
                'response_token_offsets_raw':g['response_token_offsets_raw'], 'lexical_mask':g['lexical_mask'],
                'context_token_positions':g['context_token_positions'], 'original_prefix_ids':g['input_ids'][:first],
                'original_input_ids':g['input_ids'], 'original_answer_positions':g['answer_token_positions']}
            assert len(g['input_ids']) <= limit
            if partition != 'fit':
                titles = {d['title'].strip().casefold() for d in row['documents']}
                texts = {digest(d['text']) for d in row['documents']}
                material_hash = digest(row['retrieved_passages'])
                eligible = [d for d in donor_list if d['group_id'] != plan['group_id']
                    and d['material_hash'] != material_hash and not titles.intersection(d['titles'])
                    and not texts.intersection(d['texts'])]
                eligible.sort(key=lambda d:(digest([SEED, material_hash, d['material_hash']]),d['material_hash']))
                assert eligible
                assert len(row['document_char_spans']) == 1
                a,b = row['document_char_spans'][0]
                assert row['prompt_prefix'][a:b] == row['retrieved_passages']
                selected = None
                for donor in eligible:
                    replacement = donor['material']
                    other_row = dict(row, prompt_prefix=row['prompt_prefix'][:a]+replacement+row['prompt_prefix'][b:],
                                     document_char_spans=[[a,a+len(replacement)]])
                    other = common.native.geometry(tok, other_row)
                    if len(other['input_ids']) > limit:
                        continue  # Fixed geometry-only feasibility condition, never score based.
                    for name in ('answer_token_ids','response_token_offsets','response_token_offsets_raw'):
                        assert other[name] == g[name], (rid, name, 'donor changed answer coordinates')
                    first_other = other['answer_token_positions'][0]
                    plan.update(random_prefix_ids=other['input_ids'][:first_other], random_input_ids=other['input_ids'],
                        random_answer_positions=other['answer_token_positions'], donor_material_hash=donor['material_hash'],
                        donor_response_id=donor['response_id'], donor_group_id=donor['group_id'])
                    selected = donor
                    break
                assert selected is not None, (rid, 'no length-feasible disjoint fit donor')
                lumina.validate_input(plan)
                maxima['random'] = max(maxima['random'],len(plan['random_input_ids']))
            maxima['original'] = max(maxima['original'],len(plan['original_input_ids']))
            maxima['answer_tokens'] = max(maxima['answer_tokens'],len(plan['answer_token_ids']))
            plans.append(plan)
        summaries[partition] = {'answers':len(inputs[partition]), 'maxima':maxima,
            'raw_answer_tokens':sum(len(p['answer_token_ids']) for p in plans if p['partition']==partition)}
    assert sha(COPY_HEADS) == EXPECTED_HEAD_SHA
    config = {'version':'same-domain-redeep-lumina-v1','seed':SEED,'source_files':bound_files(),
        'code_sha256':{str(p):sha(p) for p in SOURCES},'candidate_heads_sha256':sha(COPY_HEADS),
        'candidate_heads':sorted(load_json(COPY_HEADS)), 'counts':COUNTS, 'summary':summaries,
        'model':str(redeep.MODEL_PATH.resolve()), 'model_revision':'351844e75ed0bcbbe3f10671b3c808d2b83894ee',
        'redeep':{'formula':'paper standard JSD and retrieved-document ECS; code formula retained only diagnostic',
            'k_heads':1,'k_layers':10,'alpha':1.,'beta':.2,'selection':'all raw fit-token Pearson, native token risk-inverse ECS; no window pooling',
            'normalization':'fit-token minmax without clipping','precision':'full FP16 with CPU offload',
            'logit_chunk_tokens':1024,'native_answer_pooling':'all response tokens arithmetic mean'},
        'lumina':{'official_commit':lumina.port.OFFICIAL_COMMIT,'lambda':.5,'kernel':'cosine','topk':100,
            'precision':'NF4 doublequant BF16, FP32 signal reductions','partitions':['validation','test'],
            'donor_selection':'fit-only unique material; SHA(seed,target_material,donor_material); disjoint group/titles/texts; context-length gate',
            'native_answer_pooling':'all response tokens arithmetic mean'},
        'input_convention':'exact shared rendered prompt+answer token IDs; no author re-templating/truncation/extra space',
        'test_labels_read':False, 'prepared_gpu_used':False}
    if (HERE/'CONFIG.json').exists():
        previous = load_json(HERE/'CONFIG.json'); prev_hash=previous.pop('plans_sha256')
        assert previous == config, 'Prepared definition changed; preserve old outputs and use a new version'
        assert sha(HERE/'plans.jsonl') == prev_hash
    else:
        write_jsonl(HERE/'plans.jsonl',plans)
        config['plans_sha256'] = sha(HERE/'plans.jsonl'); publish(HERE/'CONFIG.json',config)
    print(json.dumps({'prepared':True, 'summary':summaries, 'GPU_used':False}))


def load_config():
    config = load_json(HERE/'CONFIG.json')
    assert config['source_files'] == bound_files(), 'Dataset provenance drift'
    for path, expected in config['code_sha256'].items():
        assert sha(path) == expected, ('Code changed after preparation',path)
    assert sha(HERE/'plans.jsonl') == config['plans_sha256']
    assert sha(COPY_HEADS) == config['candidate_heads_sha256']
    return config


def npz_path(method, plan):
    return HERE/method/'features'/plan['partition']/(plan['response_id']+'.npz')


def cache_valid(method, plan, config):
    path = npz_path(method,plan); receipt = path.with_suffix('.json')
    if not path.exists() and not receipt.exists(): return False
    assert path.exists() and receipt.exists(), ('Partial cache pair requires inspection',path)
    meta = load_json(receipt)
    assert meta['npz_sha256'] == sha(path) and meta['config_sha256'] == sha(HERE/'CONFIG.json')
    assert meta['plan_sha256'] == digest(plan)
    with np.load(path,allow_pickle=False) as z:
        n=len(plan['answer_token_ids'])
        assert np.array_equal(z['token_ids'],plan['answer_token_ids'])
        assert np.array_equal(z['offsets'],plan['response_token_offsets'])
        assert np.array_equal(z['predictor_positions'],np.asarray(plan['original_answer_positions'])-1)
        shapes = {k:(n,32) for k in ['ecs_code','ecs_paper','pks_code','pks_paper']} if method=='redeep' else {'signals':(n,7)}
        for key,shape in shapes.items(): assert z[key].shape==shape and np.isfinite(z[key]).all(), (path,key)
    return True


def save_cache(method,plan,arrays,seconds):
    path=npz_path(method,plan);path.parent.mkdir(parents=True,exist_ok=True)
    arrays.update(token_ids=np.asarray(plan['answer_token_ids'],np.int64),offsets=np.asarray(plan['response_token_offsets'],np.int32),
        predictor_positions=np.asarray(plan['original_answer_positions'],np.int64)-1)
    pending=path.with_suffix('.npz.pending')
    with pending.open('wb') as out:np.savez_compressed(out,**arrays)
    pending.replace(path)
    publish(path.with_suffix('.json'),{'response_id':plan['response_id'],'npz_sha256':sha(path),
        'config_sha256':sha(HERE/'CONFIG.json'),'plan_sha256':digest(plan),'seconds':seconds,
        'test_labels_read':False,'all_answer_tokens_preserved':True})


def verify_model():
    manifest=load_json(BENCH/'model_download_manifest.json')
    assets={}
    for item in manifest['files']:
        path=redeep.MODEL_PATH/item['filename'];actual=sha(path)
        assert actual==item['actual_sha256'],path
        assets[item['filename']]=actual
    return assets


def extract(method,limit=None):
    import torch
    from transformers import AutoModelForCausalLM
    from accelerate import cpu_offload
    config=load_config();plans=readl(HERE/'plans.jsonl')
    plans=[p for p in plans if method=='redeep' or p['partition']!='fit']
    pending=[p for p in plans if not cache_valid(method,p,config)]
    target=HERE/method;target.mkdir(parents=True,exist_ok=True)
    if not pending:
        publish(target/'EXTRACTION_COMPLETE.json',{'status':'complete','records':len(plans),'config_sha256':sha(HERE/'CONFIG.json'),
            'test_labels_read':False,'all_records_validated':True});return
    assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assets=verify_model();started=time.perf_counter();device=torch.device('cuda:0')
    capture=None
    # Only this explicit command acquires CUDA resources. Shared coordinator owns scheduling.
    with redeep.exclusive_gpu_lock('same-domain-extension-'+method):
        if method=='redeep':
            model=AutoModelForCausalLM.from_pretrained(redeep.MODEL_PATH,torch_dtype=torch.float16,
                low_cpu_mem_usage=True,attn_implementation='eager',local_files_only=True).eval()
            head=model.lm_head.weight.detach().clone().to(device)
            norm=model.model.norm.weight.detach().clone().to(device)
            cpu_offload(model,execution_device=device,offload_buffers=True)
            capture=redeep.ReDeEPActivationCapture(model,config['candidate_heads'])
            runtime={'precision':'FP16 CPU offload','projection_chunk_tokens':1024}
        else:
            model,runtime=common.load_nf4()
        publish(target/'RUNTIME.json',{'runtime':runtime,'model_assets':assets,'GPU':torch.cuda.get_device_name(0),
            'software':{name:importlib.metadata.version(name) for name in ['torch','transformers','accelerate','numpy']},
            'config_sha256':sha(HERE/'CONFIG.json')})
        try:
            with torch.inference_mode():
                for index,plan in enumerate(pending):
                    tick=time.perf_counter()
                    if method=='redeep':
                        capture.reset(len(plan['original_prefix_ids']),len(plan['original_input_ids']),plan['context_token_positions'])
                        ids=torch.tensor([plan['original_input_ids']],device=device)
                        model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                                    output_attentions=False,output_hidden_states=False,return_dict=True)
                        ec,ep=capture.ecs(device)
                        pc,pp=redeep.project_pks(capture,head,norm,float(model.config.rms_norm_eps),1024)
                        arrays={'ecs_code':ec,'ecs_paper':ep,'pks_code':pc,'pks_paper':pp}
                    else:
                        values,meta=lumina.extract(model,plan);arrays={'signals':values.cpu().numpy()}
                    torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                    save_cache(method,plan,arrays,elapsed)
                    assert cache_valid(method,plan,config)
                    progress={'method':method,'complete':len(plans)-len(pending)+index+1,'total':len(plans),
                        'last':plan['response_id'],'last_seconds':elapsed,'elapsed_seconds':time.perf_counter()-started}
                    publish(target/'PROGRESS.json',progress)
                    if index%10==0:print(json.dumps(progress),flush=True)
                    if limit is not None and index+1>=limit:break
        finally:
            if capture is not None:capture.close()
    completed=sum(cache_valid(method,p,config) for p in plans)
    if completed==len(plans):
        publish(target/'EXTRACTION_COMPLETE.json',{'status':'complete','records':completed,'config_sha256':sha(HERE/'CONFIG.json'),
            'elapsed_seconds':time.perf_counter()-started,'test_labels_read':False,'all_records_validated':True})
    print(json.dumps({'method':method,'complete':completed,'required':len(plans)}),flush=True)


def fit_redeep(plans,config):
    fit=[p for p in plans if p['partition']=='fit']
    seqs={r['response_id']:r for r in readl(RUN/'data/fit/sequences.jsonl')}
    ecs,pks,labels=[],[],[]
    for plan in fit:
        seq=seqs[plan['response_id']];y=np.asarray(seq['risk_mask'],bool)
        assert np.array_equal(seq['response_token_ids'],plan['answer_token_ids'])
        with np.load(npz_path('redeep',plan)) as z:
            ecs.append(z['ecs_paper'].astype(np.float64))
            pks.append(z['pks_paper'].astype(np.float64))
            labels.extend(y.astype(np.int8).tolist())
    e,p,y=np.concatenate(ecs),np.concatenate(pks),np.asarray(labels,np.float64)
    def correlations(x,target):
        centered=x-x.mean(0);t=target-target.mean()
        denom=np.sqrt((centered*centered).sum(0)*(t*t).sum())
        assert (denom>0).all(), 'Undefined Pearson on a constant candidate feature; inspect before selecting'
        return (centered*t[:,None]).sum(0)/denom
    er,pr=correlations(e,1-y),correlations(p,y)
    heads=config['candidate_heads']
    selected_heads=sorted(range(32),key=lambda i:(-er[i],tuple(heads[i])))[:1]
    selected_layers=sorted(range(32),key=lambda i:(-pr[i],i))[:10]
    ev,pv=[],[]
    for plan in fit:
        with np.load(npz_path('redeep',plan)) as z:
            ev.append(z['ecs_paper'][:,selected_heads].astype(np.float64).sum(1))
            pv.append(z['pks_paper'][:,selected_layers].astype(np.float64).sum(1))
    result={'selected_head_indices':selected_heads,'selected_heads':[heads[i] for i in selected_heads],
        'selected_layers':selected_layers,'head_correlations':er.tolist(),'layer_correlations':pr.tolist(),
        'ecs_bounds':formula.stable_minmax_fit(np.concatenate(ev)),
        'pks_bounds':formula.stable_minmax_fit(np.concatenate(pv)), 'fit_answers':len(fit),'fit_tokens':len(y),
        'fit_labels_sha256':sha(RUN/'data/fit/sequences.jsonl'),'config_sha256':sha(HERE/'CONFIG.json'),
        'formula':'PKS_minmax - 0.2 ECS_minmax; fit only, no clipping','validation_or_test_labels_used':False}
    assert result['ecs_bounds'][1]>result['ecs_bounds'][0] and result['pks_bounds'][1]>result['pks_bounds'][0]
    publish(HERE/'redeep/FIT.json',result)
    return result


def score(method):
    config=load_config();plans=readl(HERE/'plans.jsonl')
    relevant=[p for p in plans if method=='redeep' or p['partition']!='fit']
    assert all(cache_valid(method,p,config) for p in relevant), 'Incomplete raw extraction'
    fit=fit_redeep(plans,config) if method=='redeep' else None
    outputs=[]
    for partition in ('validation','test'):
        predictions=[];sources={}
        for plan in relevant:
            if plan['partition']!=partition:continue
            path=npz_path(method,plan)
            with np.load(path,allow_pickle=False) as z:
                if method=='lumina':risk=z['signals'][:,2].astype(np.float64)
                else:
                    e=z['ecs_paper'][:,fit['selected_head_indices']].astype(np.float64).sum(1)
                    p=z['pks_paper'][:,fit['selected_layers']].astype(np.float64).sum(1)
                    risk=formula.stable_minmax_apply(p,fit['pks_bounds'])-.2*formula.stable_minmax_apply(e,fit['ecs_bounds'])
            assert risk.shape==(len(plan['answer_token_ids']),) and np.isfinite(risk).all()
            predictions.append({'response_id':plan['response_id'],'token_scores':risk.tolist(),
                'native_answer_score':float(risk.mean()),'response_token_offsets':plan['response_token_offsets'],
                'input_sha256':plan['input_sha256'],'raw_features_sha256':sha(path)})
            sources[str(path)]=sha(path)
        assert len(predictions)==COUNTS[partition]
        out=HERE/method/f'{partition}_predictions.jsonl';write_jsonl(out,predictions)
        receipt={'method':method,'partition':partition,'records':len(predictions),'predictions_sha256':sha(out),
            'config_sha256':sha(HERE/'CONFIG.json'),'features':sources,'test_labels_read':False,
            'score_direction':'higher is risk','native_answer_pooling':'mean of all original answer tokens',
            'same_answer_tokens_and_offsets':True,'thresholds_fitted':False}
        if fit:receipt['fit_sha256']=sha(HERE/'redeep/FIT.json')
        publish(out.with_suffix('.receipt.json'),receipt);outputs.append(str(out))
    publish(HERE/method/'PREDICTIONS_COMPLETE.json',{'status':'complete','outputs':outputs,'test_labels_read':False})
    print(json.dumps({'method':method,'outputs':outputs}))


def cpu_check():
    import torch
    assert not torch.cuda.is_initialized()
    config=load_config()
    import selftest_redeep_formal_baseline_v1 as check
    check.OUT=HERE/'REDEEP_CPU_CHECK.json'
    check.main()
    report=lumina.cpu_selfcheck();publish(HERE/'LUMINA_CPU_CHECK.json',report)
    from sklearn.preprocessing import MinMaxScaler
    fit_values=np.asarray([-2.,.4,7.],np.float64)
    outside=np.asarray([-3.,0.,2.,10.],np.float64)
    scaler=MinMaxScaler().fit(fit_values[:,None])
    expected=scaler.transform(outside[:,None]).ravel()
    actual=formula.stable_minmax_apply(outside,formula.stable_minmax_fit(fit_values))
    assert np.allclose(actual,expected,rtol=0,atol=1e-15)
    assert not torch.cuda.is_initialized()
    publish(HERE/'CPU_CHECK.json',{'passed':True,'config_sha256':sha(HERE/'CONFIG.json'),
        'redeep_dense_oracle':True,'lumina_dense_oracle':True,'all_original_geometry_retokenized':True,
        'last_token_included':True,'donor_answer_token_boundaries_exact':True,
        'minmax_unclipped_sklearn_equivalence_nonconstant':True,'GPU_used':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','cpu-check','extract','score'])
    parser.add_argument('--method',choices=['redeep','lumina']);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='cpu-check':cpu_check()
    else:
        assert args.method,'--method required'
        if args.command=='extract':extract(args.method,args.limit)
        else:score(args.method)
