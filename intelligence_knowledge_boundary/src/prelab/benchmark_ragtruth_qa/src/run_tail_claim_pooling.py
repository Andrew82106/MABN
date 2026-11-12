"""Frozen-epoch, same-budget claim propagation for frozen0 and tail2.

CPU postprocessing only. Original QA fit634/cal159, never official test.
Neither model can be read for selection until its complete.json is present.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import time
import numpy as np
import torch
import run_development as q
import run_claim_pooling as pool

OUT = q.ROOT/'results/tail_claim_pooling_v1'
UPSTREAM = q.ROOT/'results/minicheck_tail_all_docs_v3'
METHODS = ('frozen0','tail2')
ALPHAS = (0,.25,.5,.75,1)

def now():return datetime.now(timezone.utc).isoformat()

def protocol():
    paths=[Path(__file__),Path(pool.__file__),Path(q.__file__),pool.PLANS,q.DATA/'gold_manifest.json',
           UPSTREAM/'protocol.json',UPSTREAM/'protocol_freeze.json']
    return {'version':'qa-tail-claim-pooling-v1','methods':list(METHODS),'alpha':list(ALPHAS),
        'scope':'CPU postprocessing of completed runs; original native QA fit634 subset and calibration159 only. Official test sealed.',
        'upstream_training':'Each original model was trained on3680 answers. Postprocessing fit results here cover only its original native634 training subset, never the complete fit set.',
        'epoch_rule':'Require both complete.json records with epochs0..3; use each original selected epoch1..3 before considering alpha. No cross-epoch/alpha joint search.',
        'geometry':'Reuse run_claim_pooling.geometry unchanged. Automatic claim boundaries from frozen plans; lexical rawBPE assigned to greatest alphanumeric overlap, earlier-claim ties. Original 4rawBPE stride1 windows and human gold unchanged.',
        'formula':'Reuse smooth: p_new=p+alpha*(claim_max-p) on lexical tokens; nonlexical tokens untouched.',
        'invariants':['alpha0 exactly reproduces original calibration42241 window and159 answer scores, thresholds, metrics.',
                      'Every alpha preserves each answer maximum and original answer threshold exactly.'],
        'selection':'For each fixed-epoch model, choose window threshold only on calibration, ties F1/precision/higher threshold; select alpha using original q.selection_key then smaller alpha. Five candidates for both weak frozen0 and tail2.',
        'reporting':'All10 candidates retained; primary report calibration only, original native_fit634_subset metrics separately labelled. Already-used calibration selection is optimistic, not independent test improvement.',
        'upstream_complete_binding':'Complete/selected epoch and prediction hashes recorded at run start, only after both complete records exist.',
        'offline':True,'GPU_used':False,'source_sha256':{str(p.resolve()):q.sha(p) for p in paths}}

def initialize():
    assert not torch.cuda.is_initialized()
    assert not (OUT/'protocol.json').exists(),'Preserve existing initialization'
    OUT.mkdir(parents=True,exist_ok=True)
    q.save(OUT/'PREFLIGHT.json',pool.self_test())
    q.save(OUT/'protocol.json',protocol())
    q.save(OUT/'STATUS.json',{'status':'frozen_waiting_for_both_completed_runs','utc':now(),
                            'official_test_opened':False,'GPU_used':False})

def selected_source(method):
    folder=UPSTREAM/method
    complete=q.read(folder/'complete.json')
    assert complete['status']=='complete_development_only' and not complete['test_opened']
    assert [e['epoch'] for e in complete['all_epochs']]==[0,1,2,3]
    chosen=complete['selected']
    assert chosen['epoch'] in (1,2,3)
    assert chosen==max(complete['all_epochs'][1:],key=lambda e:e['selection_key'])
    stem=f'epoch_{chosen["epoch"]:02d}'
    entry=q.read(folder/(stem+'.json'));assert entry==chosen
    tp=folder/(stem+'_token_predictions.npz');wp=folder/(stem+'_scores.npz')
    assert q.sha(tp)==chosen['artifacts_sha256']['_token_predictions.npz']
    assert q.sha(wp)==chosen['artifacts_sha256']['_scores.npz']
    return chosen,tp,wp,{'complete_sha256':q.sha(folder/'complete.json'),'selected_epoch':chosen['epoch'],
                       'selected_entry_sha256':q.sha(folder/(stem+'.json')),
                       'token_predictions_sha256':q.sha(tp),'original_scores_sha256':q.sha(wp)}

def run():
    assert not torch.cuda.is_initialized()
    cfg=q.read(OUT/'protocol.json');assert cfg==protocol()
    assert not (OUT/'started.json').exists(),'Do not overwrite a started comparison'
    assert all((UPSTREAM/m/'complete.json').exists() for m in METHODS),'Wait for both complete records'
    sources={m:selected_source(m) for m in METHODS}
    binding={m:sources[m][3] for m in METHODS}
    q.save(OUT/'started.json',{'utc':now(),'protocol_sha256':q.sha(OUT/'protocol.json'),'sources':binding})
    start=time.perf_counter();meta=q.metadata();groups,windows,n,_unused=pool.geometry(meta)
    assert len(meta['answers'])==793 and len(groups)==213159
    lo,hi=meta['bounds']['calibration'];assert hi-lo==42241
    fi=[i for i,a in enumerate(meta['answers']) if a['partition']=='fit']
    ci=[i for i,a in enumerate(meta['answers']) if a['partition']=='calibration']
    assert len(fi)==634 and len(ci)==159
    q.save(OUT/'geometry.json',{'raw_tokens':len(groups),'eligible_windows':len(windows),
        'calibration_windows':hi-lo,'calibration_answers':len(ci),'native_fit_subset_windows':lo,
        'native_fit_subset_answers':len(fi),'all_upstream_training_answers':3680,
        'lexical_tokens':int((groups>=0).sum()),'automatic_claims':n,'risk_gold_used_to_assign_groups':False,
        'answer_order':[a['response_id'] for a in meta['answers']]})
    families={};selected={};files=[]
    for method in METHODS:
        original,tp,wp,_=sources[method]
        with np.load(tp,allow_pickle=False) as z:
            assert len(z.files)==3839
            segments=[]
            for answer in meta['answers']:
                rid=answer['response_id'];v=z[rid]
                assert v.ndim==1 and len(v)==meta['by_response'][rid]['tokens']['token_count']
                assert np.isfinite(v).all() and np.all((v>=0)&(v<=1))
                segments.append(v)
        original_tokens=np.concatenate(segments).astype(np.float64)
        original_windows=np.asarray([original_tokens[ix].max() for ix in windows])
        original_answers=q.answer_scores(meta,original_windows)
        with np.load(wp,allow_pickle=False) as z:
            assert np.array_equal(original_windows[lo:hi],z['cal_window_scores'])
            assert np.array_equal(original_answers[ci],z['cal_answer_scores'])
            assert np.array_equal([w['label'] for w in meta['windows'][lo:hi]],z['cal_window_labels'])
            assert np.array_equal([meta['answers'][i]['label'] for i in ci],z['cal_answer_labels'])
        original_m=q.metrics(meta,original_windows,original['thresholds'])
        assert original_m['calibration']==original['calibration']
        entries=[]
        for alpha in ALPHAS:
            token=pool.smooth(original_tokens,groups,n,alpha)
            scores=np.asarray([token[ix].max() for ix in windows])
            answers=q.answer_scores(meta,scores)
            assert np.array_equal(answers,original_answers)
            ts={'window':q.choose_threshold([w['label'] for w in meta['windows'][lo:hi]],scores[lo:hi]),
                'answer':original['thresholds']['answer']}
            m=q.metrics(meta,scores,ts)
            assert m['calibration']['answers']==original['calibration']['answers']
            assert m['fit']['answers']==original_m['fit']['answers']
            if alpha==0:
                assert np.array_equal(scores,original_windows)
                assert np.array_equal(token,original_tokens)
                assert ts==original['thresholds']
                assert m['calibration']==original['calibration']
            name=f'{method}_alpha{alpha:g}'
            np.savez_compressed(OUT/(name+'_scores.npz'),token_scores=token,window_scores=scores,answer_scores=answers)
            entry={'method':method,'candidate':name,'fixed_original_selected_epoch':original['epoch'],'alpha':alpha,
                   'thresholds':ts,'selection_key':list(q.selection_key(ts,alpha)),
                   'calibration':m['calibration'],'native_fit634_subset_at_cal_thresholds':m['fit'],
                   'full3680_fit_metrics_reported':False,'all793_answer_maxima_exact':True,
                   'calibration_alpha0_original_exact':True if alpha==0 else None,
                   'scores_sha256':q.sha(OUT/(name+'_scores.npz'))}
            q.save(OUT/(name+'_result.json'),entry);entries.append(entry)
            files.extend([name+'_scores.npz',name+'_result.json'])
        families[method]=entries
        selected[method]=max(entries,key=lambda e:e['selection_key'])
        s=selected[method]
        print('TAIL_CLAIM_POOLING',method,'epoch',s['fixed_original_selected_epoch'],'alpha',s['alpha'],
              'cal_window_F1',s['calibration']['windows']['f1'],'cal_answer_F1',s['calibration']['answers']['f1'],flush=True)
    assert cfg==protocol()
    assert binding=={m:selected_source(m)[3] for m in METHODS}
    summary={'status':'complete_calibration_postprocessing_only','selected':selected,'all_candidates':families,
             'original_selected_epoch_fixed_first':True,'all_answer_scores_preserved':True,
             'scope':'cal159 primary; original native fit634 training subset descriptive only',
             'calibration_selection_optimistic':True,'official_test_opened':False,'GPU_used':False,
             'seconds':time.perf_counter()-start,'sources':binding}
    q.save(OUT/'summary.json',summary)
    report=['# 同陈述风险传播对照','',
       '仅对已完成模型的分数做 CPU 后处理。先固定各自原训练选中的轮次，再分别比较相同的 5 个 alpha；不联合重选轮次。',
       '', '| 模型 | 原轮次 | alpha | 4-BPE窗口P | R | F1 | 整答F1 |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        for e in families[method]:
            w=e['calibration']['windows'];a=e['calibration']['answers']
            label=method+('（校准选中）' if e['alpha']==selected[method]['alpha'] else '')
            report.append(f"| {label} | {e['fixed_original_selected_epoch']} | {e['alpha']:g} | {w['precision']:.4f} | {w['recall']:.4f} | {w['f1']:.4f} | {a['f1']:.4f} |")
    report += ['', '所有行均为原 159 答、42,241 个窗口的校准成绩，alpha0 逐分数、阈值和指标精确复现原结果。'
       '每答最大分数、原整答阈值均保持不变。正常回答误报没有被排除。',
       '', '原 4 个 BPE 的窗口与人工标签完全不变，自动陈述边界不使用金标。传播可以扩大高亮范围，'
       '不是得到更细的原生定位；其窗口收益或损失必须按同一分母比较。',
       '', '机器结果另存原生 fit634 训练子集，不能当作全部 3,680 条训练成绩。'
       '两种模型均获得同样 5 档后处理预算；测试仍封存，重复校准选参的涨分不代表独立测试提升。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    q.save(OUT/'STATUS.json',{'status':'complete','utc':now(),'GPU_used':False,'official_test_opened':False})
    files+=['protocol.json','PREFLIGHT.json','started.json','geometry.json','summary.json','REPORT.md','STATUS.json']
    q.save(OUT/'complete.json',{'status':'complete','utc':now(),'files_sha256':{f:q.sha(OUT/f) for f in files},
                              'sources':binding,'official_test_opened':False,'GPU_used':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['initialize','run','wait-run']);a=p.parse_args()
    if a.stage=='initialize':initialize()
    else:
        if a.stage=='wait-run':
            while not all((UPSTREAM/m/'complete.json').exists() for m in METHODS):
                print('WAIT_FOR_BOTH_COMPLETED_RUNS',flush=True);time.sleep(25)
        run()
