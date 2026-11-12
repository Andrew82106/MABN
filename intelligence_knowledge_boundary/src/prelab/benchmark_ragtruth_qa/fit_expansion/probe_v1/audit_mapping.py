"""Independent all3046 MiniCheck source binding and token mapping/PCA audit.
No production imports, fit/SVD/model/tokenizer/GPU/test. Annotation fields are
discarded immediately. Original793 window/model audits belong to parent.
"""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,importlib.util,traceback
import numpy as np
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent;E=OUT.parent;ROOT=E.parent;SEM=E/'minicheck';OLD=ROOT/'results/semantic_hidden_v1'
HELPER=ROOT/'results/development_v1/audit_coefficients_qa.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='a7a25ec0c346ad32621769762ff97b7f81c7caa6d959658a88508fb4ab5c1dae'
sp=importlib.util.spec_from_file_location('independent_inert_pickle_reader',HELPER);q=importlib.util.module_from_spec(sp);sp.loader.exec_module(q)
sha,read=q.sha,q.read;REPORT=OUT/'INDEPENDENT_MAPPING_AUDIT.json'
def save(r):REPORT.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(v):return hashlib.sha256((v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))).encode()).hexdigest()
def lines(p):
    with Path(p).open(encoding='utf-8') as f:
        for s in f:
            if s.strip():yield json.loads(s)
def independent_map(text,offsets,H,left,right,claims,support):
    # Sweep interval boundaries; average states of active token owners once
    # per constant-owner interval. Every nonwhite character has unit mass.
    events=defaultdict(lambda:[[],[]])
    for i,(l,r) in enumerate(zip(left,right)):
        assert 0<=l<r<=len(text);events[int(l)][0].append(i);events[int(r)][1].append(i)
    cuts=sorted(set(events)|{0,len(text)});active=set();chars=np.zeros((len(text),1024),np.float64)
    duplicated=0;missing=0
    for j,l in enumerate(cuts[:-1]):
        active.difference_update(events[l][1]);active.update(events[l][0]);r=cuts[j+1]
        nonwhite=sum(not c.isspace() for c in text[l:r])
        if active:
            ix=sorted(active);value=np.add.reduce(H[ix].astype(np.float64),axis=0)/len(ix);chars[l:r]=value
            duplicated+=nonwhite*(len(active)>1)
        else:missing+=nonwhite
    assert missing==0,('uncovered answer characters',missing)
    char_risk=np.zeros(len(text),np.float64);covered=np.zeros(len(text),bool)
    for c,s in zip(claims,support):
        l,r=c['start'],c['end'];assert 0<=l<r<=len(text) and text[l:r]==c['text']
        covered[l:r]=True;char_risk[l:r]=np.maximum(char_risk[l:r],1-float(s))
    mapped=np.zeros((len(offsets),1024),np.float32);risk=np.zeros(len(offsets),np.float64);lex=[];white=0
    for j,(l,r) in enumerate(offsets):
        assert 0<=l<r<=len(text)
        nw=[i for i in range(l,r) if not text[i].isspace()]
        if nw:mapped[j]=(np.add.reduce(chars[nw],axis=0)/len(nw)).astype(np.float32)
        else:white+=1
        alnum=[i for i in range(l,r) if text[i].isalnum()];lex.append(bool(alnum))
        if alnum:
            assert covered[alnum].all();risk[j]=char_risk[alnum].max()
    return mapped,risk,lex,{'nonwhitespace_characters':sum(not c.isspace() for c in text),'duplicated_character_coverage':duplicated,'whitespace_only_raw_tokens_zeroed':white}

def audit(report):
    complete=read(OUT/'complete.json');prep=read(OUT/'preparation_complete.json');cfg=read(OUT/'protocol.json')
    assert not complete['official_test_opened'] and not prep['official_test_opened']
    assert cfg['script_sha256']==sha(E/'run_probe_expansion.py') and cfg['mapping_code_sha256']==sha(ROOT/'src/run_semantic_hidden.py')
    assert cfg['old_pca_sha256']==sha(OLD/'hidden_pca.pkl')=='63a851f006b8d91bfdd88fb4ba9c1671b14902a1f244e870948bbc78c966432d'
    for p,h in prep['source_snapshot']['files_sha256'].items():assert sha(p)==h
    for name in ['token_index.json','matrices/token_hidden64.npy','matrices/token_risk.npy']:assert sha(OUT/name)==prep['files_sha256'][name]
    assert sha(OUT/'preparation_complete.json')==complete['files_sha256']['preparation_complete.json']
    exported=read(E/'data/export_freeze.json');paths=['data/new_fit.jsonl','data/new_token_plans.jsonl','data/tokens_fit.jsonl','minicheck/new_plans.jsonl']
    for name in paths:assert sha(E/name)==exported['output_files_sha256'][str((E/name).resolve())]
    inv=E/'llama_replay_preparation/INVENTORY_REVIEW.json'
    assert sha(inv)=='5ad8d495e73e47aad582ac89ed50dd1478f5ec3376bf8681d61ae3fa2dd58b0d' and read(inv)['status']=='passed'
    visible={}
    keep=['response_id','source_id','group_id','partition','official_split','original_response','retrieved_passages','answer_sha256']
    for raw in lines(E/'data/new_fit.jsonl'):
        v={k:raw[k] for k in keep};del raw
        assert v['partition']=='fit' and v['official_split']=='train' and v['response_id'] not in visible
        visible[v['response_id']]=v
    tp={p['response_id']:p for p in lines(E/'data/new_token_plans.jsonl')}
    plans={p['response_id']:p for p in lines(SEM/'new_plans.jsonl')}
    toks={}
    for raw in lines(E/'data/tokens_fit.jsonl'):
        rid=raw['response_id']
        if rid in visible:toks[rid]={k:raw[k] for k in ['response_id','source_id','group_id','partition','token_count','token_ids','answer_token_positions','response_token_offsets','response_token_offsets_raw','lexical_mask']}
        del raw
    fm=read(SEM/'claim_feature_manifest.json');done=read(SEM/'inference_complete.json');snap=read(SEM/'inference_source_snapshot.json')
    assert fm['status']==done['status']=='complete' and fm['answers']==done['rows']==3046 and not fm['test_opened'] and not done['test_opened']
    assert sha(SEM/'claim_feature_manifest.json')==done['claim_feature_manifest_sha256']
    assert fm['source_snapshot_sha256']==done['source_snapshot_sha256']==digest(snap)
    for path,key in [('data/export_freeze.json','export_freeze_sha256'),('run_minicheck_expansion.py','runner_sha256'),('protocol.json','protocol_sha256'),('minicheck/new_plans.jsonl','new_plans_sha256'),('minicheck/reuse_manifest.json','old_reuse_manifest_sha256')]:
        assert sha(E/path)==snap[key]
    assert snap['device']=='cuda:0' and snap['dtype']=='float32' and snap['no_training_or_test']
    records={r['response_id']:r for r in fm['records']};scores={r['response_id']:r for r in done['records']}
    assert len(records)==len(scores)==len(visible)==3046 and set(records)==set(scores)==set(visible)==set(tp)==set(plans)==set(toks)
    full_index=read(OUT/'token_index.json')['answers'];assert len(full_index)==3839
    index={x['response_id']:x for x in full_index};assert len(index)==3839
    cursor=0
    for j,x in enumerate(full_index):
        assert x['left']==cursor and x['right']>cursor and x['partition']==('fit' if j<3680 else 'calibration');cursor=x['right']
    assert cursor==708506==prep['total_tokens'] and set(x['response_id'] for x in full_index[634:3680])==set(visible)
    projected=np.load(OUT/'matrices/token_hidden64.npy',mmap_mode='r',allow_pickle=False)
    stored_risk=np.load(OUT/'matrices/token_risk.npy',mmap_mode='r',allow_pickle=False)
    assert projected.shape==(708506,64) and projected.dtype==np.float32 and stored_risk.shape==(708506,) and stored_risk.dtype==np.float64
    # Independently derive original fit-only PCA sampling identities/weights;
    # only use its frozen mean/components for forward projection.
    old_index=read(OLD/'token_index.json')['answers'];assert len(old_index)==793
    assert all(x['partition']=='fit' for x in old_index[:634]) and all(x['partition']=='calibration' for x in old_index[634:])
    old_ids={x['response_id'] for x in old_index};fit_ids={x['response_id'] for x in old_index[:634]};cal_ids={x['response_id'] for x in old_index[634:]}
    assert not old_ids&set(visible)
    white={r['existing_response_id']:r for r in lines(E/'fit_source_whitelist.jsonl')};assert set(white)==fit_ids
    na=Counter(x['group_id'] for x in old_index[:634]);assert len(na)==615
    pca=q.unpickle(OLD/'hidden_pca.pkl');assert (pca['fit_answers'],pca['fit_groups'],pca['sample_count'])==(634,615,20288)
    assert pca['mean'].shape==(1024,) and pca['components'].shape==(64,1024) and pca['mean'].dtype==pca['components'].dtype==np.float64
    expected=[];weights=[]
    for x in old_index[:634]:
        n=x['right']-x['left'];sample=np.floor(np.linspace(0,n-1,min(n,32))).astype(np.int64)
        for t in sample:
            expected.append({'response_id':x['response_id'],'group_id':x['group_id'],'token_index':int(t)});weights.append(1/(na[x['group_id']]*len(sample)))
    assert expected==pca['sample']
    w=np.asarray(weights,np.float64);w/=w.sum();assert np.array_equal(w,pca['sample_weights'])
    assert not {x['response_id'] for x in pca['sample']}&(cal_ids|set(visible))
    assert pca['seed']==20260924 and pca['n_iter']==3 and pca['whiten'] is False and np.isfinite(pca['components']).all()
    report['PCA']={'frozen_sha256':sha(OLD/'hidden_pca.pkl'),'fit_answers':634,'fit_groups':615,'sample_count':20288,
      'all_sample_identities_and_group_answer_balanced_weights_exact':True,'calibration_and_auxiliary_sample_count':0,
      'fit_recomputed':False,'projection':'Use frozen mean/components; float64 center/product then float32.'}
    counts=Counter();max_projection=0.;max_risk=0.
    for i,rid in enumerate(plans):
        v=visible[rid];p=plans[rid];t=toks[rid];token_plan=tp[rid];rec=records[rid];ix=index[rid]
        for key in ['source_id','group_id','partition']:
            assert t[key]==v[key]==token_plan[key]
        assert p['partition']==ix['partition']=='fit' and p['group_id']==ix['group_id']==v['group_id']
        assert token_plan['labels_used'] is False and token_plan['official_split']=='train'
        assert p['answer_sha256']==v['answer_sha256']==token_plan['answer_sha256']==digest(v['original_response'])
        assert p['document_sha256']==digest(v['retrieved_passages']) and p['truncated_input_tokens']==0 and p['uncovered_nonwhitespace_chars']==0
        assert len(p['pair_lengths'])==len(p['claims'])*len(p['document_chunks']) and max(p['pair_lengths'])<=512
        text=v['original_response'];doc=v['retrieved_passages'];doccover=np.zeros(len(doc),bool)
        for ch in p['document_chunks']:
            l,r=ch['start'],ch['end'];assert 0<=l<r<=len(doc) and doc[l:r]==ch['text'];doccover[l:r]=True
        assert all(doccover[j] or c.isspace() for j,c in enumerate(doc))
        for k in ['answer_token_positions','response_token_offsets','response_token_offsets_raw']:assert t[k]==token_plan['original'][k]
        assert t['token_ids']==token_plan['original']['answer_token_ids']
        n=t['token_count'];assert n==len(t['token_ids'])==ix['right']-ix['left']
        path=SEM/'claim_features'/f'{rid}.npz';sidepath=path.with_suffix('.json');scorepath=SEM/'scores'/f'{rid}.json'
        side=read(sidepath);score=read(scorepath)
        assert sha(path)==rec['npz_sha256']==side['npz_sha256'] and sha(sidepath)==rec['metadata_sha256']
        assert sha(scorepath)==scores[rid]['file_sha256']==side['score_row_sha256']
        assert side['response_id']==score['response_id']==rid and side['partition']==score['partition']=='fit'
        assert side['plan_sha256']==score['plan_sha256']==digest(p) and side['source_snapshot_sha256']==score['source_snapshot_sha256']==digest(snap)
        assert side['claims']==p['claims'] and side['original_answer_sha256']==p['answer_sha256']
        assert side['model_revision']=='74c8919647e61ed0f71bc177d94f10930f090068' and side['hidden_dimension']==1024 and side['dtype']=='float32' and side['uncovered_nonwhitespace_chars']==0
        assert score['device']=='cuda:0' and score['truncated_input_tokens']==0
        with np.load(path,allow_pickle=False) as z:arr={k:z[k] for k in z.files}
        expected_keys={'hidden_last','token_ids','token_start','token_end','input_token_index','claim_index','document_index','selected_document_per_claim','selected_support_per_claim'}
        assert set(arr)==expected_keys
        H=arr['hidden_last'];assert H.shape==(rec['tokens'],1024) and H.dtype==np.float32 and np.isfinite(H).all() and rec['tokens']==side['tokens']
        for k in ['token_ids','token_start','token_end','input_token_index','claim_index','document_index']:assert arr[k].shape==(len(H),)
        matrix=np.asarray(score['support_by_claim_document'],np.float32);assert matrix.shape==(len(p['claims']),len(p['document_chunks'])) and np.isfinite(matrix).all()
        assert np.all((matrix>=0)&(matrix<=1))
        choice=np.argmax(matrix,axis=1);support=matrix[np.arange(len(choice)),choice]
        assert np.array_equal(choice,arr['selected_document_per_claim']) and choice.tolist()==side['selected_document_per_claim']
        assert np.array_equal(support,arr['selected_support_per_claim']) and np.array_equal(support,np.asarray(score['max_support']))
        assert np.array_equal(1-support.astype(np.float64),np.asarray(score['claim_risk']))
        assert arr['selected_support_per_claim'].dtype==np.float32
        for ci,c in enumerate(p['claims']):
            mask=arr['claim_index']==ci;assert mask.any()
            assert np.all(arr['document_index'][mask]==choice[ci]) and np.all(arr['token_start'][mask]>=c['start']) and np.all(arr['token_end'][mask]<=c['end'])
            assert np.all(np.diff(arr['input_token_index'][mask])>0)
        assert np.array_equal(np.unique(arr['claim_index']),np.arange(len(p['claims'])))
        mapped,risk,lex,stat=independent_map(text,t['response_token_offsets'],H,arr['token_start'],arr['token_end'],p['claims'],support)
        assert lex==t['lexical_mask']
        got=((mapped.astype(np.float64)-pca['mean'])@pca['components'].T).astype(np.float32)
        target=projected[ix['left']:ix['right']];targetrisk=stored_risk[ix['left']:ix['right']]
        er=float(np.max(np.abs(got-target)));rr=float(np.max(np.abs(risk-targetrisk)));max_projection=max(max_projection,er);max_risk=max(max_risk,rr)
        assert np.array_equal(got,target),('projection',rid,er)
        assert np.array_equal(risk,targetrisk),('risk',rid,rr)
        counts.update(stat);counts.update(answers=1,raw_llama_tokens=n,selected_minicheck_tokens=len(H),claims=len(p['claims']),document_chunks=len(p['document_chunks']),pairs=len(p['pair_lengths']))
        if (i+1)%250==0 or i==3045:
            report.update(coverage=dict(counts),max_projection_abs=max_projection,max_token_risk_abs=max_risk);save(report)
            print('EXPANSION_MAPPING_PASSED',i+1,'rawtokens',counts['raw_llama_tokens'],flush=True)
    assert counts['answers']==3046 and counts['raw_llama_tokens']==495347 and counts['selected_minicheck_tokens']==fm['tokens']==417082
    assert read(OUT/'preparation_complete.json')==prep and sha(OLD/'hidden_pca.pkl')==cfg['old_pca_sha256']
    report.update(status='passed',blockers=[],coverage=dict(counts),max_projection_abs=max_projection,max_token_risk_abs=max_risk,
      all3046_claim_cache_manifest_sidecar_score_plan_answer_partition_bindings_exact=True,
      all3046_argmax_selected_document_support_risk_consistent=True,all495347_token_projection_and_token_risk_exact=True,
      all_new_raw_and_clipped_coordinates_and_lexical_masks_exact=True,
      existing_pca_mean_components_unchanged=True,full_output_token_index_contiguous_and_correct=True,
      bytes_hashes={'probe_complete':sha(OUT/'complete.json'),'probe_preparation':sha(OUT/'preparation_complete.json'),'claim_feature_manifest':sha(SEM/'claim_feature_manifest.json'),
      'inference_complete':sha(SEM/'inference_complete.json'),'token_hidden64':sha(OUT/'matrices/token_hidden64.npy'),'token_risk':sha(OUT/'matrices/token_risk.npy'),
      'inventory_review':sha(inv),'audit_helper':sha(__file__)})
    md=['新增映射独立审计通过，无阻断。全部3046条的claim缓存、sidecar、score、plan、原答案/文档哈希和fit身份绑定一致；最大支持度选文档及首个并列规则一致。','',
      f"独立按字符区间重做映射，再用原PCA64投影：全部{counts['raw_llama_tokens']}个Llama词元的64维状态及token risk逐项精确一致。{counts['selected_minicheck_tokens']}个MiniCheck词元覆盖全部非空白答案字符，重叠词元先按字符平均；风险仅由实际字母/数字字符相交的claim决定。",'',
      '旧PCA文件未变，20288个采样身份及权重全部仅来自原634 fit回答/615组；无新增回答或calibration进入PCA样本。本次没有重新拟合PCA，也未重算其SVD。','',
      '本审计不重复窗口聚合或LR结果，也不加载模型、运行GPU、读取官方test/withheld或检查人工标签值。该状态来自额外的双向MiniCheck、可见完整claim；不是原生成器的因果白盒trace。']
    (OUT/'INDEPENDENT_MAPPING_AUDIT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')

if __name__=='__main__':
    report={'status':'running','audit_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/data_build/extract_review',
      'no_new_fit_SVD_or_GPU':True,'production_imports':False,'official_test_or_withheld_contents_read':False,'annotation_values_inspected':False,
      'frozen_files_modified':False,'limits':['Source/cache consistency and CPU mapping/projection, not a replay of MiniCheck model inference.',
      'PCA fitting provenance is checked by frozen-file binding and allfit sample identities/weights; no SVD is rerun.',
      'Original793 reuse, window aggregation, detector fits and thresholds are reviewed separately by parent.']}
    try:
        with threadpool_limits(limits=4):audit(report)
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());save(report);raise
    save(report);print('EXPANSION_MAPPING_AUDIT_PASSED',sha(REPORT),flush=True)

