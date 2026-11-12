"""Independent text-only grouping and frozen postprocessing replay; no fitting."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,time
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];DATA=ROOT/'data'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def lines(p):
    with Path(p).open(encoding='utf-8') as f:
        for s in f:
            if s.strip():yield json.loads(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def threshold(y,s):
    y=np.asarray(y,int);s=np.asarray(s,np.float64);ix=np.argsort(-s,kind='stable');ss=s[ix];yy=y[ix]
    last=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1];tp=np.r_[0,np.cumsum(yy)[last]];n=np.r_[0,last+1]
    cut=np.r_[np.nextafter(ss[0],np.inf),ss[last]];f1=2*tp/(n+y.sum());pr=np.divide(tp,n,out=np.zeros(len(n)),where=n>0)
    j=max(range(len(n)),key=lambda j:(f1[j],pr[j],cut[j]))
    return {'threshold':float(cut[j]),'f1':float(f1[j]),'precision':float(pr[j]),'rows':len(y),'positive':int(y.sum())}
def metrics(y,s,t):
    y=np.asarray(y,int);p=s>=t;tp=int(((y==1)&p).sum());fp=int(((y==0)&p).sum());fn=int(((y==1)&~p).sum());tn=int(((y==0)&~p).sum())
    return {'n':len(y),'positive':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'auroc':float(roc_auc_score(y,s)),'average_precision':float(average_precision_score(y,s))}
def main():
    started=time.perf_counter();cfg=read(OUT/'protocol.json');summary=read(OUT/'summary.json')
    assert sha(ROOT/'src/run_claim_pooling.py')==cfg['script_sha256']
    for p,h in read(OUT/'complete.json')['files_sha256'].items():assert sha(OUT/p)==h
    pp=ROOT/'semantic_baseline/cuda_variant/plans.jsonl';assert sha(pp)==cfg['plans_sha256']
    plans={r['response_id']:r for r in lines(pp)}
    # Group assignment uses only text/offsets/claims. Official risk spans are never consulted here.
    parts=('fit','calibration');answers=[];tokens=[];windows=[];bounds={}
    for part in parts:
        aa=list(lines(DATA/f'answers_{part}.jsonl'));tt=list(lines(DATA/f'tokens_{part}.jsonl'));ww=list(lines(DATA/f'windows_k4_{part}.jsonl'))
        assert [a['response_id'] for a in aa]==[t['response_id'] for t in tt]
        bounds[part]=(len(windows),len(windows)+len(ww));answers+=aa;tokens+=tt;windows+=ww
    assert len(answers)==len(tokens)==793 and len(windows)==210364
    groups=[];starts={};group_start=0;empty_claims=0;bytoken={};answer_token_ranges=[];ties=0;cross_claim_tokens=0
    for a,t in zip(answers,tokens):
        rid=a['response_id'];text=t['original_response'];claims=plans[rid]['claims'];offsets=t['response_token_offsets']
        alnum=np.array([c.isalnum() for c in text]);cs=np.r_[0,np.cumsum(alnum)];covered=np.zeros(len(text),bool)
        for c in claims:covered[c['start']:c['end']]=True
        assert np.all(covered[alnum])
        g=np.full(len(offsets),-1,int)
        for i,(lo,hi) in enumerate(offsets):
            lexical=cs[hi]>cs[lo];assert bool(lexical)==bool(t['lexical_mask'][i])
            if not lexical:continue
            overlap=[int(cs[min(hi,c['end'])]-cs[max(lo,c['start'])]) if min(hi,c['end'])>max(lo,c['start']) else 0 for c in claims]
            maximum=max(overlap);assert maximum>0
            best=overlap.index(maximum);g[i]=group_start+best
            ties+=int(sum(x==maximum for x in overlap)>1);cross_claim_tokens+=int(sum(x>0 for x in overlap)>1)
        empty_claims+=len(claims)-len(set(g[g>=0].tolist()));starts[rid]=len(groups)
        answer_token_ranges.append((len(groups),len(groups)+len(g)))
        groups.extend(g);group_start+=len(claims);bytoken[rid]=t
    groups=np.asarray(groups);valid=groups>=0
    assert len(groups)==213159 and valid.sum()==174518 and group_start==8852
    assert empty_claims==7 and len(set(groups[valid]))==8845
    win=np.full((len(windows),4),-1,int);aw=defaultdict(list)
    for i,w in enumerate(windows):
        rid=w['response_id'];ix=np.array(w['token_indices'])+starts[rid];lex=ix[groups[ix]>=0]
        assert len(lex)>0;win[i,:len(lex)]=lex;aw[rid].append(i)
        assert w['label']==int(any(bytoken[rid]['risk_mask'][j] for j in w['token_indices']))
    widvalid=win>=0;wx=np.maximum(win,0);metricerror=0.;candidatecount=0;max_checks=0;results={};nwindowexact=0;ntokenexact=0
    for method,source in cfg['methods'].items():
        folder=ROOT/'results'/source;assert sha(folder/'complete.json')==cfg['upstream_complete_sha256'][source]
        old=read(folder/'summary.json')['selected'][method];sp=folder/method/f"epoch_{old['epoch']:03d}_scores.npz"
        with np.load(sp,allow_pickle=False) as z:raw={k:z[k].copy() for k in ('token_scores','window_scores','answer_scores')}
        v=raw['token_scores'].astype(np.float64);members=defaultdict(list)
        for i in np.flatnonzero(valid):members[int(groups[i])].append(int(i))
        maxima={k:max(v[ix]) for k,ix in members.items()};permax=np.array([maxima[int(g)] if g>=0 else v[i] for i,g in enumerate(groups)])
        entries=[]
        for alpha in cfg['alpha']:
            n=v.copy();n[valid]=v[valid]+alpha*(permax[valid]-v[valid]);assert np.array_equal(n[~valid],v[~valid])
            for group,ix in members.items():assert max(n[ix])==maxima[group]
            for lo,hi in answer_token_ranges:assert max(n[lo:hi])==max(v[lo:hi]);max_checks+=1
            gathered=n[wx];gathered[~widvalid]=-np.inf;ws=gathered.max(axis=1)
            ans=np.array([ws[aw[a['response_id']]].max() for a in answers]);assert np.array_equal(ans,raw['answer_scores'])
            name=f'{method}_alpha{alpha:g}';e=read(OUT/(name+'_result.json'))
            assert e['source_scores_sha256']==sha(sp) and e['scores_sha256']==sha(OUT/(name+'_scores.npz'))
            with np.load(OUT/(name+'_scores.npz'),allow_pickle=False) as z:
                assert np.array_equal(n,z['token_scores']) and np.array_equal(ws,z['window_scores']) and np.array_equal(ans,z['answer_scores'])
            ntokenexact+=len(n);nwindowexact+=len(ws)
            if alpha==0:assert np.array_equal(ws,raw['window_scores']) and e['metrics']==old['metrics'] and e['thresholds']==old['thresholds']
            lo,hi=bounds['calibration'];th=threshold([w['label'] for w in windows[lo:hi]],ws[lo:hi]);assert th==e['thresholds']['window']
            assert e['thresholds']['answer']==old['thresholds']['answer']
            key=[min(th['f1'],old['thresholds']['answer']['f1']),th['f1'],th['precision'],-alpha]
            assert key==e['selection_key'];entries.append(e);candidatecount+=1
            for part in parts:
                lo,hi=bounds[part];ai=np.array([i for i,a in enumerate(answers) if a['partition']==part])
                for unit,y,s,tt in [('windows',[w['label'] for w in windows[lo:hi]],ws[lo:hi],th['threshold']),('answers',[answers[i]['label'] for i in ai],ans[ai],old['thresholds']['answer']['threshold'])]:
                    m=metrics(y,s,tt)
                    for k,x in m.items():
                        delta=abs(x-e['metrics'][part][unit][k]);assert delta<=5e-15,(name,part,unit,k,delta);metricerror=max(metricerror,delta)
                    if unit=='answers':assert e['metrics'][part][unit]==old['metrics'][part][unit]
        chosen=max(entries,key=lambda e:e['selection_key']);assert chosen==summary['selected'][method]
        results[method]={'alpha':chosen['alpha'],'window_F1':chosen['metrics']['calibration']['windows']['f1'],'answer_F1':chosen['metrics']['calibration']['answers']['f1']}
    assert candidatecount==20 and max_checks==15860
    result={'status':'passed','blockers':[],'no_new_fit_or_GPU':True,'official_test_opened':False,
      'geometry':{'raw_tokens':213159,'lexical_tokens':174518,'automatic_claims':8852,'assigned_claim_groups':8845,'claims_without_lexical_tokens':7,'answers':793,'windows':210364,'cross_claim_tokens':cross_claim_tokens,'greatest_overlap_ties':ties,'text_only_independent_reconstruction':True,'label_values_not_used_in_grouping':True,'each_lexical_token_one_group_and_no_cross_answer_group':True},
      'all20_token_arrays_exact':ntokenexact,'all20_window_arrays_exact':nwindowexact,'all20_answer_arrays_exact':15860,'each_claim_and_answer_raw_max_unchanged':True,'answer_raw_max_checks':max_checks,
      'calibration_window_thresholds_exact':20,'original_answer_thresholds_unchanged':20,'identity_full_metrics_exact':4,'selection_keys_exact':20,'alpha_choices_exact':4,'metric_blocks':80,'metrics_max_abs':metricerror,
      'results':results,'interpretation':'Within existing automatic claim propagation is offline and may broaden a local alert to other facts in a claim; fixed raw4BPE gold stays unchanged. Entire-answer metrics are invariant by construction, not an additional improvement.',
      'limits':['Repeated calibration selection; not independent test improvement.','Original code reads gold spans only to write descriptive extent diagnostics; those values do not feed assignment, token scores, filtering, or alpha candidates.','Upstream model training not reaudited here; source completion and selected score hashes verified.'],
      'seconds':time.perf_counter()-started,'files_sha256':{str(p.resolve()):sha(p) for p in [Path(__file__),ROOT/'src/run_claim_pooling.py',pp,OUT/'protocol.json',OUT/'geometry.json',OUT/'complete.json',OUT/'summary.json']}}
    (OUT/'INDEPENDENT_AUDIT_CLAIM_POOLING.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'INDEPENDENT_AUDIT_CLAIM_POOLING.md').write_text('独立审计通过。按原文本和自动陈述边界重建 174,518 个有字母或数字的词元归组；没有用风险标签选组。\n\n20 个候选的词元、窗口、整答分数均精确复现，20 个校准窗口阈值和 4 个 alpha 选择一致。每个陈述及每份回答的最大值保持，原整答阈值及成绩完全不变；alpha=0 完整复现旧成绩。\n\n这是离线传播分数：可能同时扩大陈述内其他正常事实的报警范围。原 4 BPE 标签未改，涨分仅为重复校准集探索，不能算独立测试提升。上游训练未重复审查，来源完成文件和选中分数哈希已核对。\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','all20_token_arrays_exact','all20_window_arrays_exact','answer_raw_max_checks','metrics_max_abs','results','seconds')}))
if __name__=='__main__':main()
