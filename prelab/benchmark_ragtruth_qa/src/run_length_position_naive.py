"""Two fixed, positive-direction length/position scores; no model fitting."""
from pathlib import Path
import os
import time
import numpy as np
from threadpoolctl import threadpool_limits
import run_development as q

OUT=q.ROOT/'results/length_position_naive_v1'


def protocol():
    return {'version':'qa-length-position-naive-v1','scope':'Original793 answers:634fit/159calibration,210364 unchanged raw4BPE windows. No official test.',
        'scores':{'answer_length':'log1p(raw_token_count) of the answer, broadcast to all its eligible windows',
                  'absolute_window_end':'log1p(window.token_end), original exclusive raw-BPE endpoint; no length normalization'},
        'direction':'Higher score means higher risk, fixed before scoring. No sign search or fitted parameters.',
        'thresholds':'Original q.choose_threshold separately for window and whole-answer on159cal only: max F1, then precision, then higher threshold.',
        'answer_aggregation':'Original maximum across every eligible window, including normal answers and safe refusals retained by official good QA gold.',
        'reporting':'All two scores fit/cal results; no score-family winner selected. Fit is descriptive, not in-sample learned performance.',
        'motivation':'Findings EMNLP2025.952 studies spurious task correlations and distribution generalization; it does not prove this project has a length shortcut.',
        'paper':'https://aclanthology.org/2025.findings-emnlp.952/',
        'no_factual_input':True,'trained':False,'GPU_used':False,'official_test_opened':False}


def direct_counts(y, pred):
    tp=fp=fn=tn=0
    for truth,guess in zip(y,pred):
        if truth and guess:tp+=1
        elif guess:fp+=1
        elif truth:fn+=1
        else:tn+=1
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def run():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'started.json').exists()
    cfg=protocol();q.save(OUT/'protocol.json',cfg)
    snap={str(p.resolve()):q.sha(p) for p in (Path(__file__),Path(q.__file__),q.DATA/'gold_manifest.json')}
    q.save(OUT/'started.json',{'pid':os.getpid(),'time':time.time(),'protocol_sha256':q.sha(OUT/'protocol.json'),'source_sha256':snap,'trained':False})
    start=time.perf_counter();meta=q.metadata()
    assert meta['bounds']=={'fit':[0,168123],'calibration':[168123,210364]} and len(meta['answers'])==793
    wc=len(meta['windows']);ac=len(meta['answers']);window_ids=np.asarray([w['window_id'] for w in meta['windows']]);answer_ids=np.asarray([a['response_id'] for a in meta['answers']])
    ys=np.asarray([w['label'] for w in meta['windows']],int);ya=np.asarray([a['label'] for a in meta['answers']],int)
    length=np.log1p(np.asarray([meta['by_response'][w['response_id']]['tokens']['token_count'] for w in meta['windows']],np.float64))
    position=np.log1p(np.asarray([w['token_end'] for w in meta['windows']],np.float64))
    assert length.shape==position.shape==(210364,) and np.all(position<=length)
    lo,hi=meta['bounds']['calibration'];results={};audit={};files=[]
    for name,s in [('answer_length',length),('absolute_window_end',position)]:
        a=q.answer_scores(meta,s)
        threshold={'window':q.choose_threshold(ys[lo:hi],s[lo:hi]),'answer':q.choose_threshold(ya[634:],a[634:])}
        m=q.metrics(meta,s,threshold);wp=s>=threshold['window']['threshold'];ap=a>=threshold['answer']['threshold']
        check={}
        for part,(l,r) in meta['bounds'].items():
            ai=np.asarray([j for j,x in enumerate(meta['answers']) if x['partition']==part],int)
            check[part]={'windows':direct_counts(ys[l:r],wp[l:r]),'answers':direct_counts(ya[ai],ap[ai])}
            for level in ['windows','answers']:
                for k,v in check[part][level].items():assert m[part][level][k]==v,(name,part,level,k)
        path=OUT/f'{name}_predictions.npz'
        np.savez_compressed(path,window_scores=s,answer_scores=a,window_predictions=wp,answer_predictions=ap,
                            window_ids=window_ids,answer_ids=answer_ids,window_gold=ys,answer_gold=ya)
        entry={'thresholds':threshold,'metrics':m,'prediction_sha256':q.sha(path),'trained_parameters':0,'score_is_probability':False}
        q.save(OUT/f'{name}_result.json',entry);results[name]=entry;audit[name]=check;files += [path.name,f'{name}_result.json']
        print('NAIVE_COMPLETE',name,'cal_window',m['calibration']['windows']['f1'],'cal_answer',m['calibration']['answers']['f1'],flush=True)
    with np.load(OUT/'answer_length_predictions.npz',allow_pickle=False) as z1,np.load(OUT/'absolute_window_end_predictions.npz',allow_pickle=False) as z2:
        same_answer=bool(np.array_equal(z1['answer_scores'],z2['answer_scores']))
        different_answer_count=int(np.count_nonzero(z1['answer_scores']!=z2['answer_scores']))
    q.save(OUT/'COUNTS_CHECK.json',{'passed':True,'direct_boolean_counts':audit,'answer_scores_same':same_answer,'answer_scores_differ_count':different_answer_count})
    q.save(OUT/'summary.json',{'methods':results,'answers':ac,'windows':wc,'bounds':meta['bounds'],'window_order_sha256':q.digest(window_ids.tolist()),
        'answer_order_sha256':q.digest(answer_ids.tolist()),'seconds':time.perf_counter()-start,'trained':False,'GPU_used':False,'official_test_opened':False})
    report=['# 长度与位置朴素对照','',
        '只使用回答词元数或窗口末端位置，不读取事实语义，不拟合参数；固定高分为高风险。两方法独立在原cal选窗口及整答阈值。', '',
        '| 分数 | 划分 | 窗口P | 窗口R | 窗口F1 | 窗口AUROC | 整答P | 整答R | 整答F1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,e in results.items():
        for part in q.PARTITIONS:
            w=e['metrics'][part]['windows'];a=e['metrics'][part]['answers']
            report.append(f"| {name} | {part} | {w['precision']:.6f} | {w['recall']:.6f} | {w['f1']:.6f} | {w['auroc']:.6f} | {a['precision']:.6f} | {a['recall']:.6f} | {a['f1']:.6f} |")
    report += ['','保留全部793个原答和210364个原4BPE窗口；fit634/168123窗，cal159/42241窗。原窗口标点规则、人工标注和全窗口max完全不变。逐项预测及阈值、TP/FP/FN/TN、AP/AUROC均保存。',
        f'两方法整答max分数完全相同：{same_answer}；不同答案数：{different_answer_count}。末端位置取max通常就是答案长度，不能将两者当作两个独立整答检测信号。',
        'cal中100/159答案有风险；全答报风险的整答F1已为200/259≈0.772201。整答F1接近该值不能单独说明可靠。',
        '论文动机是检查伪相关和泛化问题，其主要发现涉及跨任务类别；没有证明本项目存在长度/位置捷径。这里仅是本项目的开发对照，不能据此修改标签或作因果归因。[官方论文](https://aclanthology.org/2025.findings-emnlp.952/)']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n','utf-8')
    assert cfg==protocol()
    for path,h in snap.items():assert q.sha(path)==h
    files += ['protocol.json','started.json','COUNTS_CHECK.json','summary.json','REPORT.md']
    q.save(OUT/'complete.json',{'status':'complete','files_sha256':{p:q.sha(OUT/p) for p in files},'trained':False,'GPU_used':False,'official_test_opened':False})
    print('NAIVE_ALL_COMPLETE',round(time.perf_counter()-start,2),'seconds',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
