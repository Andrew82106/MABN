"""Independent fixed persistence replay in log-odds space; no fit/GPU/test.

Only approved fit/calibration exports and selected frozen development scores.
No production module is imported or executed. Writes only new audit artifacts.
"""
from pathlib import Path
import importlib.util
from datetime import datetime, timezone
import hashlib
import json
import traceback
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PRIOR=OUT.parent/'development_v1'
REPORT=OUT/'INDEPENDENT_AUDIT_PERSISTENCE_QA.json'
HELPER=PRIOR/'audit_coefficients_qa.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='a7a25ec0c346ad32621769762ff97b7f81c7caa6d959658a88508fb4ab5c1dae'
spec=importlib.util.spec_from_file_location('prior_independent_metadata',HELPER)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
read,sha,near=q.read,q.sha,q.near
NFIT,NCAL=168123,42241


def write(report):
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def independent_chains(answers,windows,indices):
    chains=[];gaps=[]
    for answer in answers:
        ix=indices[answer['answer_id']]
        starts=[windows[j]['token_indices'][0] for j in ix]
        assert np.all(np.diff(starts)>0)
        chain=[]
        for j,start in zip(ix,starts):
            if chain and start!=windows[chain[-1]]['token_indices'][0]+1:
                gaps.append({'answer_id':answer['answer_id'],'partition':answer['partition'],
                             'previous_start':windows[chain[-1]]['token_indices'][0],'next_start':start})
                chains.append(np.asarray(chain,int));chain=[]
            chain.append(j)
        assert chain;chains.append(np.asarray(chain,int))
    assert np.array_equal(np.sort(np.concatenate(chains)),np.arange(NFIT+NCAL))
    for c in chains:
        assert len({windows[j]['answer_id'] for j in c})==1
        assert len({windows[j]['partition'] for j in c})==1
        assert np.all(np.diff([windows[j]['token_indices'][0] for j in c])==1)
    assert len(chains)==len(answers)+len(gaps)
    padded=np.full((len(chains),max(map(len,chains))),-1,np.int64)
    for j,c in enumerate(chains):padded[j,:len(c)]=c
    return chains,gaps,padded


def log_odds_smoothing(scores,padded,stay,temperature):
    assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    if stay==.5 and temperature==1.:return scores.copy()
    clipped=np.clip(scores,1e-12,1-1e-12)
    emission=(np.log(clipped)-np.log1p(-clipped))/temperature
    ls,ld=np.log(stay),np.log1p(-stay)
    def transition(odds):
        return np.logaddexp(ls+odds,ld)-np.logaddexp(ld+odds,ls)
    forward=np.empty_like(scores);backward=np.zeros_like(scores)
    forward[padded[:,0]]=emission[padded[:,0]]
    for depth in range(1,padded.shape[1]):
        keep=padded[:,depth]>=0
        now=padded[keep,depth];previous=padded[keep,depth-1]
        forward[now]=emission[now]+transition(forward[previous])
    for depth in range(padded.shape[1]-2,-1,-1):
        keep=padded[:,depth+1]>=0
        now=padded[keep,depth];nxt=padded[keep,depth+1]
        backward[now]=transition(emission[nxt]+backward[nxt])
    return q.sigmoid(forward+backward)


def binary(y,s,threshold):
    y=np.asarray(y,int);p=np.asarray(s)>=threshold
    tp=int(np.count_nonzero((y==1)&p));fp=int(np.count_nonzero((y==0)&p))
    fn=int(np.count_nonzero((y==1)&~p));tn=int(np.count_nonzero((y==0)&~p))
    return {'n':len(y),'positive':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def count_compare(actual,reference,tag):
    for key in actual:
        if isinstance(actual[key],int):assert actual[key]==reference[key],(tag,key)
        else:near(actual[key],reference[key],(tag,key),atol=1e-14)


def sources(started,answers,indices):
    result={}
    for directory,binding in started['source'].items():
        assert directory in ('development_v1','harp_development_v1','sequence_v1')
        source=OUT.parent/directory
        assert sha(source/'complete.json')==binding['complete_sha256']
        assert sha(source/'summary.json')==binding['summary_sha256']
        complete,summary=read(source/'complete.json'),read(source/'summary.json')
        complete_hashes={k.replace('\\','/'):v for k,v in complete['files_sha256'].items()}
        assert len(complete_hashes)==len(complete['files_sha256'])
        for name,selected in summary['selected'].items():
            filename=(f"{name}/epoch_{selected['epoch']:03d}_scores.npz" if directory=='sequence_v1' else selected['candidate']+'_scores.npz')
            path=source/filename
            assert sha(path)==binding['scores'][filename]
            assert complete_hashes[filename]==binding['scores'][filename]
            with np.load(path,allow_pickle=False) as z:s=z['window_scores'].astype(float);a=z['answer_scores']
            assert s.shape==(NFIT+NCAL,) and a.shape==(793,) and name not in result
            independent=np.asarray([max(s[indices[x['answer_id']]]) for x in answers])
            assert np.array_equal(independent,a)
            result[name]=(s,selected)
    assert len(result)==9
    return result


def audit(report):
    complete=read(OUT/'complete.json');started=read(OUT/'started.json');summary=read(OUT/'summary.json');protocol=read(OUT/'protocol.json')
    assert complete['status']=='complete' and complete['test_opened'] is False
    for name,h in complete['files_sha256'].items():assert sha(OUT/name)==h,name
    assert started['code_sha256']==sha(ROOT/'src/run_persistence.py')
    assert started['protocol_sha256']==sha(OUT/'protocol.json')
    assert started['gold_sha256']==sha(ROOT/'data/gold_manifest.json')
    assert started['test_opened'] is False and summary['test_opened'] is False
    assert summary['calibration_results_are_selection_optimistic'] is True
    assert protocol['stay']==[.5,.8,.95,.99] and protocol['temperature']==[.5,1.,2.]
    answers,windows,indices=q.metadata()
    chains,gaps,padded=independent_chains(answers,windows,indices)
    calchains=[c for c in chains if c[0]>=NFIT]
    assert len(chains)==started['chains']==1187 and len(calchains)==started['calibration_chains']==235
    report['geometry']={'answers':793,'fit_answers':634,'calibration_answers':159,
        'eligible_windows':NFIT+NCAL,'fit_windows':NFIT,'calibration_windows':NCAL,
        'chains':len(chains),'fit_chains':len(chains)-len(calchains),'calibration_chains':len(calchains),
        'raw_start_gaps':len(gaps),'fit_gaps':sum(g['partition']=='fit' for g in gaps),'calibration_gaps':sum(g['partition']=='calibration' for g in gaps),
        'all_gaps_split':True,'all_answers_separate':True,'all_partitions_separate':True,'all_window_ids_exactly_once':True}
    # Functional isolation test: alter every later chain; first chain unchanged.
    synthetic=np.linspace(.05,.95,len(windows))
    shifted=synthetic.copy();shifted[chains[0]]=synthetic[chains[0]]
    other=np.ones(len(windows),bool);other[chains[0]]=False;shifted[other]=1-shifted[other]
    ref=log_odds_smoothing(synthetic,padded,.95,2.)
    alt=log_odds_smoothing(shifted,padded,.95,2.)
    assert np.array_equal(ref[chains[0]],alt[chains[0]])
    report['independent_chain_isolation_test']=True
    data=sources(started,answers,indices)
    assert set(data)==set(summary['selected'])==set(summary['all_candidates'])
    comparisons={}
    for name,(raw,old) in data.items():
        selected=summary['selected'][name];table=summary['all_candidates'][name]
        assert len(table)==12
        assert [(x['stay'],x['temperature']) for x in table]==[(a,b) for a in protocol['stay'] for b in protocol['temperature']]
        identity=[t for t in table if t['stay']==.5 and t['temperature']==1.]
        assert len(identity)==1 and identity[0]['thresholds']==old['thresholds']
        assert np.array_equal(log_odds_smoothing(raw,padded,.5,1.),raw)
        keys=[]
        for candidate in table:
            w,a=candidate['thresholds']['window'],candidate['thresholds']['answer']
            key=[min(w['f1'],a['f1']),w['f1'],w['precision'],int(candidate['stay']==.5 and candidate['temperature']==1.),-candidate['stay'],-abs(np.log(candidate['temperature']))]
            assert key==candidate['key'];keys.append(key)
        chosen=max(range(len(keys)),key=lambda j:keys[j])
        assert chosen==selected['selected_index']
        assert all(selected[k]==v for k,v in table[chosen].items())
        got=log_odds_smoothing(raw,padded,selected['stay'],selected['temperature'])
        with np.load(OUT/(name+'_scores.npz'),allow_pickle=False) as z:
            saved=z['window_scores'];saved_a=z['answer_scores']
        error=near(got,saved,name+' independent log-odds replay',rtol=3e-12,atol=3e-12)
        av=np.asarray([max(got[indices[a['answer_id']]]) for a in answers])
        ae=near(av,saved_a,name+' answer max',rtol=3e-12,atol=3e-12)
        assert np.array_equal(np.asarray([max(saved[indices[a['answer_id']]]) for a in answers]),saved_a)
        for part,left,right,ai in [('fit',0,NFIT,range(634)),('calibration',NFIT,NFIT+NCAL,range(634,793))]:
            ts=selected['thresholds']
            wc=binary([w['label'] for w in windows[left:right]],saved[left:right],ts['window']['threshold'])
            ac=binary([answers[j]['label'] for j in ai],saved_a[list(ai)],ts['answer']['threshold'])
            count_compare(wc,selected['metrics'][part]['windows'],(name,part,'window'))
            count_compare(ac,selected['metrics'][part]['answers'],(name,part,'answer'))
        oldcal,newcal=old['metrics']['calibration'],selected['metrics']['calibration']
        assert old['metrics']==selected['raw_metrics']
        comparisons[name]={'stay':selected['stay'],'temperature':selected['temperature'],
            'window_before':oldcal['windows']['f1'],'window_after':newcal['windows']['f1'],
            'window_delta':newcal['windows']['f1']-oldcal['windows']['f1'],
            'answer_before':oldcal['answers']['f1'],'answer_after':newcal['answers']['f1'],
            'answer_delta':newcal['answers']['f1']-oldcal['answers']['f1'],
            'full_window_score_replay_max_abs':error,'full_answer_score_replay_max_abs':ae,
            'identity_thresholds_exact':True,'all_12_selection_keys_recomputed':True}
        report['calibration_comparisons']=comparisons;write(report)
        print('PERSISTENCE_QA_SELECTED_REPLAY_PASSED',name,flush=True)
    report['outcome']={'localization_improved':sum(x['window_delta']>0 for x in comparisons.values()),
        'answers_improved':sum(x['answer_delta']>0 for x in comparisons.values()),
        'answers_declined':sum(x['answer_delta']<0 for x in comparisons.values()),
        'interpretation':'All9 selected persistence variants improve calibration window F1 and worsen calibration answer F1. The max-min key is governed by lower window F1 here; no joint answer improvement or fresh-test benefit is established.'}
    report['all_selected_replay_window_scores']=9*(NFIT+NCAL)
    report['all_selected_replay_answer_scores']=9*793
    report['source_sha256']={str(p.resolve()):sha(p) for p in (Path(__file__),HELPER,ROOT/'src/run_persistence.py',OUT/'protocol.json',OUT/'started.json',OUT/'summary.json',OUT/'complete.json',ROOT/'data/gold_manifest.json')}
    cal=OUT/'CALIBRATION_AUDIT_PERSISTENCE_QA.json'
    if cal.exists():
        cr=read(cal);assert cr['passed'] is True and cr['blockers']==[]
        report['independent_all_candidate_calibration_audit']={'path':str(cal.resolve()),'sha256':sha(cal),'status':'passed'}
    text=['# QA persistence 独立审计','',
      '9组均提高校准定位F1，但全部降低整答F1。参数由同一校准集选择，这些数值有选择乐观，不能视为最终测试改善。','',
      '| 方法 | stay / temperature | 定位F1 原→平滑 | 整答F1 原→平滑 |',
      '|---|---|---:|---:|']
    for name,x in comparisons.items():
        text.append(f"| {name} | {x['stay']} / {x['temperature']} | {x['window_before']:.6f} → {x['window_after']:.6f} | {x['answer_before']:.6f} → {x['answer_after']:.6f} |")
    text+=['','已核1187条独立链，包括394处raw-start缺口；不跨答案或fit/cal分区。9组全量冻结分数由独立log-odds前后递推重算，原identity阈值精确保留。','',
           '未重新拟合、使用GPU、读取官方test/sealed/withheld或修改冻结产物。']
    (OUT/'INDEPENDENT_AUDIT_PERSISTENCE_QA.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':
    report={'status':'running','utc':datetime.now(timezone.utc).isoformat(),'reviewer':'/root/data_build/extract_review',
        'no_refitting':True,'no_gpu':True,'official_test_or_withheld_read':False,
        'method':'Reconstruct chain boundaries from frozen raw-token indices; independent vectorized log-odds HMM forward/backward, not production probability recursion.',
        'limits':['Calibration outcomes select persistence parameters and remain selection-optimistic. No fresh-test claim.',
                  'Main replay verifies9 frozen selected settings; all108 candidates/216 thresholds are handled by separate independent calibration subaudit.',
                  'Frozen hashes and numerical consistency do not independently prove unrecorded external execution history.']}
    try:
        audit(report);report['status']='passed'
    except Exception as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc());write(report);raise
    write(report)
    print('PERSISTENCE_QA_INDEPENDENT_AUDIT_PASSED',sha(REPORT),flush=True)
