"""Text-only claim pooling that preserves every original answer maximum.

All input models are already frozen. This is a calibrated postprocessing
ablation, not another trained model or a change to the evaluation unit.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,time
import numpy as np
import run_development as q

OUT=q.ROOT/'results/claim_pooling_v1'
SOURCES={'token_tcn':'sequence_v1','full_lb_pca64_tcn':'sequence_full_v2',
         'full_lb_harp64_tcn':'sequence_full_v2','minicheck_hidden64_risk_tcn_w32':'semantic_sequence_v1'}
PLANS=q.ROOT/'semantic_baseline/cuda_variant/plans.jsonl'

def protocol():
    return {'version':'qa-claim-pooling-v1','methods':SOURCES,'alpha':[0,.25,.5,.75,1],
       'hypothesis':'Human spans sometimes mark an entire claim. Test controlled within-claim propagation without changing raw4BPE labels or answer scores.',
       'groups':'Existing automatic claims, no gold; each lexical raw token assigned to claim with greatest alphanumeric character overlap, ties earlier claim. Every lexical token covered exactly once; no cross-answer group.',
       'formula':'p_new(token)=p(token)+alpha*(max_lexical_token_p_in_its_claim-p(token)); nonlexical tokens untouched',
       'invariant':'Every answer maximum is exactly unchanged for all five alphas. Alpha0 is exact identity for all windows. Original answer threshold retained.',
       'selection':'Window threshold optimized only on calibration. Same q.selection_key then smaller alpha; no fitting, five candidates per input model.',
       'scope':'Same fit634/calibration159 only, repeated calibration selection optimistic, official test unopened',
       'offline':True,'script_sha256':q.sha(__file__),'plans_sha256':q.sha(PLANS),
       'upstream_complete_sha256':{n:q.sha(q.ROOT/'results'/n/'complete.json') for n in sorted(set(SOURCES.values()))}}

def geometry(meta):
    plans={r['response_id']:r for r in q.lines(PLANS)};groups=[];starts={};cursor=0;group_cursor=0;diagnostics=[]
    texts={r['response_id']:r['original_response'] for part in q.PARTITIONS for r in q.lines(q.DATA/(part+'.jsonl'))}
    for answer in meta['answers']:
        rid=answer['response_id'];p=plans[rid];t=meta['by_response'][rid]['tokens'];text=texts[rid]
        g=np.full(t['token_count'],-1,np.int64);claims=p['claims']
        for i,(a,b) in enumerate(t['response_token_offsets']):
            chars=[j for j in range(a,b) if text[j].isalnum()]
            assert bool(chars)==bool(t['lexical_mask'][i])
            if not chars:continue
            overlap=[sum(c['start']<=j<c['end'] for j in chars) for c in claims]
            chosen=max(range(len(claims)),key=lambda j:(overlap[j],-j));assert overlap[chosen]>0
            assert all(any(c['start']<=j<c['end'] for c in claims) for j in chars)
            g[i]=group_cursor+chosen
        starts[rid]=cursor;cursor+=len(g);group_cursor+=len(claims);groups.extend(g.tolist())
        for lab in answer['original_labels']:
            chars={j for j in range(lab['start'],lab['end']) if text[j].isalnum()}
            overlaps=[(len(chars.intersection(j for j in range(c['start'],c['end']) if text[j].isalnum())),
                       sum(text[j].isalnum() for j in range(c['start'],c['end']))) for c in claims]
            best=max((a/b if b else 0 for a,b in overlaps),default=0)
            diagnostics.append({'response_id':rid,'partition':answer['partition'],'type':lab['label_type'],
                 'span_start':lab['start'],'span_end':lab['end'],'span_chars':len(chars),'best_claim_lexical_coverage':best})
    assert cursor==213159
    windows=[];groups=np.asarray(groups)
    for w in meta['windows']:
        ix=np.asarray(w['token_indices'])+starts[w['response_id']];ix=ix[groups[ix]>=0];assert len(ix)
        windows.append(ix)
    return np.asarray(groups),windows,group_cursor,diagnostics

def smooth(scores,groups,count,alpha):
    v=np.asarray(scores,np.float64);valid=groups>=0;maxima=np.full(count,-np.inf)
    np.maximum.at(maxima,groups[valid],v[valid]);out=v.copy()
    if alpha:out[valid]=v[valid]+alpha*(maxima[groups[valid]]-v[valid])
    assert np.isfinite(out).all() and np.all(out>=v)
    oldmax=np.full(count,-np.inf);newmax=oldmax.copy()
    np.maximum.at(oldmax,groups[valid],v[valid]);np.maximum.at(newmax,groups[valid],out[valid])
    assert np.array_equal(oldmax,newmax),'Claim/answer maxima must not change'
    return out

def self_test():
    p=np.asarray([.1,.9,.3,.6,.7]);g=np.asarray([0,0,-1,1,1])
    for a in [0,.25,.5,.75,1]:
        out=smooth(p,g,2,a)
        assert out[2]==p[2] and out[1]==p[1] and out[4]==p[4]
        assert out[0]==p[0]+a*(p[1]-p[0]) and out[3]==p[3]+a*(p[4]-p[3])
    return {'passed':True,'group_max_preserved':True,'nonlexical_unchanged':True}

def run():
    cfg=q.read(OUT/'protocol.json');assert cfg==protocol();assert not (OUT/'started.json').exists()
    q.save(OUT/'started.json',{'utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':q.sha(OUT/'protocol.json')})
    start=time.perf_counter();meta=q.metadata();groups,windows,n,diag=geometry(meta)
    q.save(OUT/'geometry.json',{'raw_tokens':len(groups),'lexical_tokens':int((groups>=0).sum()),'automatic_claims':n,
       'assigned_groups':len(set(groups[groups>=0].tolist())),'eligible_windows':len(windows),'risk_labels_used_for_geometry':False})
    q.save(OUT/'span_extent_diagnostic.json',{'rows':diag,'purpose':'Descriptive only; no labels changed or used for grouping or filtering'})
    selected={};families={};files=[];lo,hi=meta['bounds']['calibration']
    for method,source in SOURCES.items():
        folder=q.ROOT/'results'/source;e=q.read(folder/'summary.json')['selected'][method]
        name=f'epoch_{e["epoch"]:03d}_scores.npz';path=folder/method/name
        with np.load(path,allow_pickle=False) as z:original={k:z[k].copy() for k in ('token_scores','window_scores','answer_scores')}
        entries=[]
        for alpha in cfg['alpha']:
            t=smooth(original['token_scores'],groups,n,alpha);scores=np.asarray([t[ix].max() for ix in windows])
            answers=q.answer_scores(meta,scores);assert np.array_equal(answers,original['answer_scores'])
            if alpha==0:assert np.array_equal(scores,original['window_scores'])
            ts={'window':q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]],scores[lo:hi]),'answer':e['thresholds']['answer']}
            m=q.metrics(meta,scores,ts)
            for part in q.PARTITIONS:assert m[part]['answers']==e['metrics'][part]['answers']
            if alpha==0:assert m==e['metrics'] and ts==e['thresholds']
            key=q.selection_key(ts,alpha);candidate=f'{method}_alpha{alpha:g}'
            np.savez_compressed(OUT/(candidate+'_scores.npz'),window_scores=scores,answer_scores=answers,token_scores=t)
            entry={'candidate':candidate,'alpha':alpha,'thresholds':ts,'selection_key':list(key),'metrics':m,
              'source_scores_sha256':q.sha(path),'scores_sha256':q.sha(OUT/(candidate+'_scores.npz')),'answer_scores_exact':True}
            q.save(OUT/(candidate+'_result.json'),entry);entries.append(entry);files.extend([candidate+'_scores.npz',candidate+'_result.json'])
        families[method]=entries;selected[method]=max(entries,key=lambda e:e['selection_key'])
        print('CLAIM_POOLING',method,selected[method]['alpha'],selected[method]['metrics']['calibration']['windows']['f1'],flush=True)
    assert cfg==protocol()
    q.save(OUT/'summary.json',{'selected':selected,'all_candidates':families,'seconds':time.perf_counter()-start,
           'all_answer_scores_preserved':True,'official_test_opened':False,'calibration_selection_optimistic':True})
    report=['在自动划分的同一陈述内传播部分风险分数，整答最大分数严格保持不变。仅开发校准结果。','',
        '| 方法 | alpha | 定位F1 | 整答F1 |','|---|---:|---:|---:|']
    for name,e in selected.items():
        m=e['metrics']['calibration'];report.append(f"| {name} | {e['alpha']:g} | {m['windows']['f1']:.4f} | {m['answers']['f1']:.4f} |")
    report+=['','四模型各5个相同候选，alpha0精确复现原指标。句子划分不使用人工标注；4原始词元评测及原标签完全不变。',
         '不同于重新训练，此方法只利用既有token分数做离线处理。测试仍封存，选参涨分不等于独立测试改进。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n','utf-8')
    files+=['protocol.json','started.json','geometry.json','span_extent_diagnostic.json','summary.json','REPORT.md']
    q.save(OUT/'complete.json',{'utc':datetime.now(timezone.utc).isoformat(),'files_sha256':{f:q.sha(OUT/f) for f in files},'official_test_opened':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','run']);a=p.parse_args()
    if a.stage=='initialize':assert not (OUT/'protocol.json').exists();q.save(OUT/'PREFLIGHT.json',self_test());q.save(OUT/'protocol.json',protocol())
    else:run()
