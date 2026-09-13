"""Fixed Llama replay on 3046 additional published fit answers; no gold interface.

prepare/cpu-check are CPU-only. Explicit run is reserved for root GPU scheduling.
All old answers, labels, features, and native/reconstruction provenance stay frozen.
"""
from pathlib import Path
import argparse,gc,json,os,shutil,sys,time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
import feature_qa as base
import run_feature_qa as loader
import feature_lookback_controls_v2 as controls

OUT=HERE/'llama_features_v2';PLANS=HERE/'data/new_token_plans.jsonl'
VERSION='qa-expanded-fit-uniform-llama-replay-v2-fixture-id';EXPECTED=3046
CONTROL_NAMES=controls.VARIANTS[:-1]
SCHEMA={'lb':'float32[N,1024], original source-only/post-read/no-header Lookback',
 'nll':'float32[N], selected published token NLL at P+j-1',
 'hidden_last':'float32[N,4096], final RMSNorm state at P+j',
 **{k:'float32[N,1024], unchanged frozen 2x2 control definition' for k in CONTROL_NAMES},
 'token_ids':'int64[N]','answer_token_positions':'int64[N]',
 'response_token_offsets':'int32[N,2]','response_token_offsets_raw':'int32[N,2]',
 'token_start':'int32[N]','token_end':'int32[N]'}

def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]
def freeze(p,value):
    assert json.loads(json.dumps(value,ensure_ascii=False))==value,'Frozen structures must be JSON-native'
    if p.exists():assert base.read(p)==value,('Frozen expansion replay changed',str(p))
    else:base.save(p,value)

def protocol():
    return {'version':VERSION,'scope':'Only the frozen3046 additional fit responses; source634/group615 unchanged. Old634 fit and159 calibration features reused read-only; no test data.',
     'model':{'repo':loader.REPO,'revision':loader.REVISION,'load_config':loader.LOAD_CONFIG},
     'replay':'One common Llama-2-7B-Chat NF4 model teacher-forces published text from five other generators. These are uniform-checkpoint representations, NOT original-generator native internal trajectories.',
     'labels_interface':'Extractor reads only frozen new_token_plans; new_fit and annotation/provenance files may be hashed but are not parsed. No labels or source-generator IDs enter the model/hooks/selection.',
     'identity':'Every new response retained, regardless of eventual error/refusal. No answer generation, retokenization changes, source edits, condition or label filtering.',
     'coordinates':'Exact frozen full-string IDs/positions and raw/clipped offsets. First boundary-crossing token retained; duplicate byte-fallback offsets retained. No independent answer tokenization.',
     'length':'All original input+answer sequences must be <=4096; longer sequences error, never truncate silently.',
     'forward':'One model.model forward with ControlHooks produces original LB plus four context/time controls; final states also feed original chunks16 NLL and float32 final hidden.',
     'source_definition_protocol_sha256':base.sha(ROOT/'lookback_controls_protocol.json'),
     'context_time_header':'Unchanged five Lookback definitions. lb equals old lb_source_post_legacy without duplicate storage. Four controls share actual chat header and retain first token.',
     'timing':'LB post-read for original; four controls as named. NLL predicts current token from preceding state. hidden_last is after reading current token/final RMSNorm.',
     'dtype':'Preserve old float32 LB/NLL/hidden_last exports; original NF4/BF16 backbone and q/k arithmetic. No hidden compression.',
     'schema':SCHEMA,'expected_new_rows':EXPECTED,'backbone_forwards_per_new_row':1,
     'cpu_gate':'Tiny Llama/Qwen2: unified single-forward arrays exact versus existing separate base/controls extraction; first token/offsets/allfiveLB retained.',
     'gpu_gate':'Two fixed old-fit engineering rows selected by total input length: base LB/NLL/hidden and five LB definitions must equal old caches exactly. Two new source-length extremes repeated exactly; no tolerance relaxation. Engineering rows are not appended to new data.',
     'cache':'Each row bound to source plan, layout, frozen source signature and array/sidecar hashes. Full3046 manifest required before training.',
     'resource_scheduling':'CPU preparation only until root assigns GPU after MiniCheck expansion/backfill. No model fitting or scoring in this script.',
     'limitations':['Cross-generator answer supervision is not native white-box probing of each generator.','Repeated development calibration and the unchanged sealed test remain separate; this prepares fit features only.']}

def validate(arrays,plan,width=1024,hidden=4096):
    assert set(arrays)==set(SCHEMA);v=plan['original'];n=len(v['answer_token_ids'])
    for k,d in [('lb',width),('nll',None),('hidden_last',hidden)]+[(k,width) for k in CONTROL_NAMES]:
        assert arrays[k].shape==((n,) if d is None else (n,d))
        assert arrays[k].dtype==np.float32 and np.isfinite(arrays[k]).all()
        if k=='lb' or k in CONTROL_NAMES:assert np.all((arrays[k]>=0)&(arrays[k]<=1))
    for k,expected,dtype in [('token_ids',v['answer_token_ids'],np.int64),('answer_token_positions',v['answer_token_positions'],np.int64),('response_token_offsets',v['response_token_offsets'],np.int32),('response_token_offsets_raw',v['response_token_offsets_raw'],np.int32)]:
        assert arrays[k].dtype==dtype and arrays[k].tolist()==expected
    assert np.array_equal(arrays['token_start'],arrays['response_token_offsets'][:,0])
    assert np.array_equal(arrays['token_end'],arrays['response_token_offsets'][:,1])

@torch.inference_mode()
def extract(model,plan,lo):
    assert plan['partition']=='fit' and plan['official_split']=='train' and not model.training
    ids,mask,pos=base.tensor_input(model,plan['original'])
    assert ids.shape[1]<=4096
    with controls.ControlHooks(model,plan,lo) as hook:
        final=model.model(input_ids=ids,attention_mask=mask,use_cache=False,output_attentions=False,output_hidden_states=False).last_hidden_state[0]
    assert len(hook.seen)==hook.spec['layers']
    arrays={k:hook.controls[k].reshape(len(pos),-1) for k in CONTROL_NAMES}
    arrays['lb']=hook.controls['lb_source_post_legacy'].reshape(len(pos),-1)
    arrays['hidden_last']=final.index_select(0,pos).float().cpu().numpy()
    nll=[]
    for pp in pos.split(base.LOGIT_BATCH):
        lp=model.lm_head(final.index_select(0,pp-1)).float().log_softmax(-1)
        nll.extend((-lp.gather(1,ids[0,pp,None])).squeeze(-1).cpu().tolist())
    arrays['nll']=np.asarray(nll,np.float32)
    v=plan['original'];off=np.asarray(v['response_token_offsets'],np.int32)
    arrays.update(token_ids=np.asarray(v['answer_token_ids'],np.int64),answer_token_positions=np.asarray(v['answer_token_positions'],np.int64),response_token_offsets=off,
                  response_token_offsets_raw=np.asarray(v['response_token_offsets_raw'],np.int32),token_start=off[:,0].copy(),token_end=off[:,1].copy())
    del final
    validate(arrays,plan,hook.spec['lb_width'],hook.spec['hidden_size'])
    return arrays

def source_state():
    exported=base.read(HERE/'data/export_freeze.json')
    for p in [PLANS,HERE/'data/new_fit.jsonl',HERE/'data/tokenizer_signature.json']:
        assert base.sha(p)==exported['output_files_sha256'][str(p.resolve())]
    plans=rows(PLANS);assert len(plans)==EXPECTED and len({p['response_id'] for p in plans})==EXPECTED
    old=rows(ROOT/'data/feature_preparation/plans.jsonl');oldfit=[p for p in old if p['partition']=='fit']
    oldids={p['response_id'] for p in old};sources={p['source_id'] for p in oldfit};groups={p['group_id'] for p in oldfit}
    assert len(oldfit)==634 and len(sources)==634 and len(groups)==615
    assert not oldids&{p['response_id'] for p in plans}
    assert {p['source_id'] for p in plans}<=sources and {p['group_id'] for p in plans}<=groups
    assert not {p['group_id'] for p in plans}&{p['group_id'] for p in old if p['partition']=='calibration'}
    for p in plans:
        assert p['partition']=='fit' and p['official_split']=='train' and p['labels_used'] is False
        assert not any(k in p for k in ('labels','original_labels','risk','gold','model'))
        assert base.digest(p['released_prompt'])==p['prompt_sha256'] and base.digest(p['original_response'])==p['answer_sha256']
        v=p['original'];assert len(v['input_ids'])<=4096 and v['attention_mask']==[1]*len(v['input_ids'])
        assert base.digest(v['input_ids'])==v['input_ids_sha256'] and base.digest(p['original_response'])==v['response_text_sha256']
        assert [v['input_ids'][i] for i in v['answer_token_positions']]==v['answer_token_ids']
    return plans,oldfit

def prepare():
    base.assert_cpu_only();OUT.mkdir(parents=True,exist_ok=True);(OUT/'features').mkdir(exist_ok=True)
    plans,oldfit=source_state();freeze(OUT/'protocol.json',protocol())
    tok=base.AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
    assert base.tokenizer_signature(tok)==base.read(HERE/'data/tokenizer_signature.json')
    layouts=[]
    for p in plans:
        lo=controls.layout(p);layouts.append(lo)
        rr=p['original']['rendered_reference_character_range'];rr=[x-len(base.WRAPPER_LEFT) for x in rr]
        assert base.encode_view(tok,p['released_prompt'],p['original_response'],rr)==p['original'],'Frozen token coordinates differ from unchanged tokenizer'
    text=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in layouts);lp=OUT/'layouts.jsonl'
    if lp.exists():assert lp.read_text('utf-8')==text
    else:lp.write_text(text,encoding='utf-8')
    files=[Path(__file__),Path(base.__file__),Path(loader.__file__),Path(controls.__file__),ROOT/'lookback_controls_protocol.json',
      ROOT/'model_download_manifest.json',ROOT/'data/feature_manifest.json',ROOT/'data/lookback_controls_v2/feature_manifest.json',
      HERE/'data/export_freeze.json',PLANS,HERE/'data/new_fit.jsonl',HERE/'data/tokenizer_signature.json',ROOT/'data/feature_preparation/plans.jsonl',lp,OUT/'protocol.json']
    sig={'version':VERSION,'files_sha256':{str(p.resolve()):base.sha(p) for p in files},'model_repo':loader.REPO,'model_revision':loader.REVISION,'load_config':loader.LOAD_CONFIG,
      'expected_rows':EXPECTED,'response_ids_in_order':[p['response_id'] for p in plans],'schema':SCHEMA,'input_to_extractor':'Only frozen token plan and derived text-only layout; no annotations or generator ID','test_read':False}
    freeze(OUT/'signature.json',sig)
    n=sum(len(p['original']['answer_token_ids']) for p in plans)
    oldplans=rows(ROOT/'data/feature_preparation/plans.jsonl')
    proxy=sum(len(p['original']['input_ids'])*len(p['original']['answer_token_ids']) for p in plans)/sum(len(p['original']['input_ids'])*len(p['original']['answer_token_ids']) for p in oldplans)
    scalar_bytes=n*((5*1024+4096+1)*4+16+16+8)
    estimate={'new_response_rows':EXPECTED,'source_count':len({p['source_id'] for p in plans}),'group_count':len({p['group_id'] for p in plans}),'raw_response_tokens':n,
       'max_input_plus_answer_tokens':max(len(p['original']['input_ids']) for p in plans),'all_original_views_reencoded_exact':True,
       'boundary_crossing_first_token_rows':sum(p['original']['response_token_offsets_raw'][0][0]<0 for p in plans),
       'uncompressed_array_bytes':scalar_bytes,'uncompressed_GiB':scalar_bytes/2**30,'reserve_disk_GiB':40,
       'measured_reference':{'responses':793,'response_tokens':213159,'fiveLB_seconds':1566.0},'attention_query_work_ratio':proxy,
       'estimated_GPU_minutes_range':[1566*proxy*1.1/60,1566*proxy*1.5/60],
       'time_estimate_limit':'Proxy from total_input_tokens*answer_tokens; includes 10-50% overhead allowance for NLL/float32 hidden export. Not a benchmark of the new cohort.',
       'gpu_started':False,'labels_values_read':False,'old_features_unchanged':True}
    freeze(OUT/'preparation.json',estimate)
    print('LLAMA_EXPANSION_PREPARED',json.dumps(estimate),flush=True)
    return plans,layouts,oldfit,sig

def cpu_check():
    from transformers import LlamaConfig,LlamaForCausalLM,Qwen2Config,Qwen2ForCausalLM
    base.assert_cpu_only();torch.set_num_threads(4);torch.manual_seed(20260912)
    result=[]
    for name,conf,klass in [('llama',LlamaConfig,LlamaForCausalLM),('qwen2',Qwen2Config,Qwen2ForCausalLM)]:
        c=conf(vocab_size=128,hidden_size=48,intermediate_size=96,num_hidden_layers=2,num_attention_heads=6,num_key_value_heads=2,max_position_embeddings=128,attention_dropout=0.,use_sliding_window=False,sliding_window=None)
        c._attn_implementation='eager';model=klass(c).cpu().eval()
        ids=[1,11,13,17,19,23,29,31,37,41,43,47,53];pos=[8,9,10,11,12]
        view={'input_ids':ids,'attention_mask':[1]*13,'answer_token_positions':pos,'answer_token_ids':[ids[i] for i in pos],
          'context_token_positions':[2,3,4],'response_token_offsets':[[i,i+1] for i in range(5)],'response_token_offsets_raw':[[-1,1]]+[[i,i+1] for i in range(1,5)]}
        p={'response_id':'synthetic','partition':'fit','official_split':'train','original':view};lo={'prefix_context_token_positions':list(range(6)),'header_token_positions':[6,7]}
        got=extract(model,p,lo);old,_=base.extract_features(model,p);allc=controls.extract(model,p,lo)
        for k in old:assert np.array_equal(old[k],got[k]),(name,k)
        for k in CONTROL_NAMES:assert np.array_equal(got[k],allc[k]),(name,k)
        assert np.array_equal(got['lb'],allc['lb_source_post_legacy'])
        result.append({'model':name,'base_LB_NLL_hidden_and_all_coords_exact':True,'all_four_controls_exact':True,'first_token_kept':True})
        del model;gc.collect()
    value={'passed':True,'code_sha256':base.sha(__file__),'protocol_sha256':base.sha(OUT/'protocol.json'),'checks':result,'GPU_initialized':torch.cuda.is_initialized(),'new_fits':0,'test_read':False}
    freeze(OUT/'cpu_selfcheck.json',value);print('LLAMA_EXPANSION_CPU_CHECK',json.dumps(value),flush=True)

def cache(plan,lo,sig):
    rid=plan['response_id'];path=OUT/'features'/(rid+'.npz');side=path.with_suffix('.json')
    if not side.exists():assert not path.exists(),('Uncommitted cache',str(path));return None
    m=base.read(side)
    assert m['response_id']==rid and m['source_id']==plan['source_id'] and m['group_id']==plan['group_id'] and m['partition']=='fit'
    assert m['signature_sha256']==base.digest(sig) and m['plan_sha256']==base.digest(plan) and m['layout_sha256']==base.digest(lo)
    assert m['arrays_sha256']==base.sha(path) and m['complete'] and not m['labels_used'] and not m['test_read']
    with np.load(path,allow_pickle=False) as z:validate({k:z[k] for k in z.files},plan)
    return m

def commit(a,p,lo,sig):
    validate(a,p);path=OUT/'features'/(p['response_id']+'.npz');loader.save_npz(path,a)
    m={'complete':True,'response_id':p['response_id'],'source_id':p['source_id'],'group_id':p['group_id'],'partition':'fit','response_tokens':len(a['token_ids']),
       'plan_sha256':base.digest(p),'layout_sha256':base.digest(lo),'signature_sha256':base.digest(sig),'arrays_sha256':base.sha(path),
       'replay_model':loader.REPO,'replay_revision':loader.REVISION,'original_generator_native_trace':False,'labels_used':False,'test_read':False}
    base.save(path.with_suffix('.json'),m);return m

def run():
    plans,layouts,oldfit,sig=prepare();check=base.read(OUT/'cpu_selfcheck.json')
    assert check['passed'] and check['code_sha256']==base.sha(__file__) and check['protocol_sha256']==base.sha(OUT/'protocol.json')
    assert shutil.disk_usage(OUT).free>40*2**30
    lock=OUT/'.gpu_runner.lock';fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    model=None;begin=time.perf_counter()
    try:
        existing=[cache(p,l,sig) for p,l in zip(plans,layouts)]
        if not all(existing):
            for f in base.read(ROOT/'model_download_manifest.json')['files']:assert base.sha(base.MODEL/f['filename'])==f['actual_sha256']
            model,_=loader.load_nf4();gate=OUT/'gpu_selfcheck.json'
            if gate.exists():assert base.read(gate)['passed'] and base.read(gate)['signature_sha256']==base.digest(sig)
            else:
                checks=[];ordered=sorted(oldfit,key=lambda p:(len(p['original']['input_ids']),p['response_id']))
                original_records={r['response_id']:r for r in base.read(ROOT/'data/feature_manifest.json')['records']}
                control_records={r['response_id']:r for r in base.read(ROOT/'data/lookback_controls_v2/feature_manifest.json')['records']}
                for p in (ordered[0],ordered[-1]):
                    got=extract(model,p,controls.layout(p));rid=p['response_id']
                    assert base.sha(ROOT/'data/features'/(rid+'.npz'))==original_records[rid]['npz_sha256']
                    assert base.sha(ROOT/'data/features'/(rid+'.json'))==original_records[rid]['metadata_sha256']
                    assert base.sha(ROOT/'data/lookback_controls_v2/features'/(rid+'.npz'))==control_records[rid]['npz_sha256']
                    assert base.sha(ROOT/'data/lookback_controls_v2/features'/(rid+'.json'))==control_records[rid]['json_sha256']
                    with np.load(ROOT/'data/features'/(rid+'.npz'),allow_pickle=False) as z:
                        for k in ('lb','nll','hidden_last'):assert np.array_equal(got[k],z[k]),('Old anchor',rid,k)
                    with np.load(ROOT/'data/lookback_controls_v2/features'/(rid+'.npz'),allow_pickle=False) as z:
                        for k in CONTROL_NAMES:assert np.array_equal(got[k],z[k]),('Old control',rid,k)
                    checks.append({'response_id':rid,'kind':'old_fit_anchor','all_fields_exact':True})
                order=sorted(range(len(plans)),key=lambda i:(len(plans[i]['original']['input_ids']),plans[i]['response_id']))
                for i in (order[0],order[-1]):
                    a=extract(model,plans[i],layouts[i]);b=extract(model,plans[i],layouts[i])
                    assert all(np.array_equal(a[k],b[k]) for k in a)
                    existing[i]=commit(a,plans[i],layouts[i],sig);checks.append({'response_id':plans[i]['response_id'],'kind':'new_repeat','all_fields_exact':True})
                freeze(gate,{'passed':True,'signature_sha256':base.digest(sig),'checks':checks,'test_read':False})
        for i,(p,l) in enumerate(zip(plans,layouts)):
            if existing[i] is None:existing[i]=commit(extract(model,p,l),p,l,sig)
            if (i+1)%20==0 or i+1==len(plans):print('LLAMA_EXPANSION',i+1,'/',EXPECTED,'seconds',round(time.perf_counter()-begin,1),flush=True)
        records=[]
        for p,l in zip(plans,layouts):
            m=cache(p,l,sig);assert m is not None;path=OUT/'features'/(p['response_id']+'.npz')
            records.append({**m,'npz':str(path),'json':str(path.with_suffix('.json')),'npz_sha256':base.sha(path),'json_sha256':base.sha(path.with_suffix('.json'))})
        for p,h in sig['files_sha256'].items():assert base.sha(p)==h
        freeze(OUT/'feature_manifest.json',{'complete':True,'completed_count':EXPECTED,'records':records,'signature_sha256':base.digest(sig),'schema':SCHEMA,'labels_used':False,'test_read':False,'native_original_generator_trace':False})
    finally:
        if model is not None:del model;gc.collect();torch.cuda.empty_cache()
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','cpu-check','run']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='cpu-check':cpu_check()
    else:run()
