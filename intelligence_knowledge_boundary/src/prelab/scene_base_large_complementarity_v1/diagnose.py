"""Contingency counts at existing thresholds; no models, fitting or new scores."""
from pathlib import Path
import hashlib
import json
import time
from collections import defaultdict
import numpy as np

OUT=Path(__file__).resolve().parent
PRE=OUT.parent
SOURCES={'base':PRE/'round29_full_hidden_scene_adaptation/results',
         'large':PRE/'round31_large_semantic_scene_adaptation/results'}
SCENE=PRE/'benchmark_ragtruth_qa/control_scene_transfer_v1'
METHOD='r26_plus_semantic'


def read(p):return json.loads(Path(p).read_text('utf-8'))
def lines(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
def identity(row):return {k:v for k,v in row.items() if k not in ('scores','predictions')}


def count(y,p):
    tn,fp,fn,tp=map(int,np.bincount(2*y+p.astype(int),minlength=4))
    return dict(n=len(y),positive=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def contingency(rows,b,l,key):
    cells={}
    for name,mask in [('both_risk',b&l),('base_only',b&~l),('large_only',~b&l),('neither_risk',~b&~l)]:
        chosen=[r for r,on in zip(rows,mask) if on]
        cells[name]={'n':len(chosen),'groups':len({r['group_id'] for r in chosen}),
            'questions':len({r['question_id'] for r in chosen}),
            'answer_rows':len({r['row_id'] for r in chosen}),
            'ids':[r[key] for r in chosen]}
    assert sum(c['n'] for c in cells.values())==len(rows)
    return {'n':len(rows),'cells':cells}


def main():
    started=time.perf_counter();assert not (OUT/'REPORT.json').exists()
    protocol=read(OUT/'protocol.json');folds=read(SCENE/'folds.json')
    owner={}
    for i,f in enumerate(folds):
        sets=[set(f[k]) for k in ('fit_groups','calibration_groups','evaluation_groups')]
        assert all(sets[j].isdisjoint(sets[k]) for j in range(3) for k in range(j+1,3))
        assert len(set.union(*sets))==278
        for group in sets[2]:assert group not in owner;owner[group]=i
    assert len(owner)==278
    snapshots={};data={};thresholds={};summary={}
    for label,folder in SOURCES.items():
        done=read(folder/'complete.json');fit=read(folder/'fit_complete.json')
        assert done['human_gold'] is False and done['status']=='complete_development_only'
        assert fit['source_sha256'][str((SCENE/'folds.json').resolve())]==sha(SCENE/'folds.json')
        snapshots[label]={'complete_sha256':sha(folder/'complete.json'),'files_sha256':{}}
        for name,h in done['files_sha256'].items():
            assert sha(folder/name)==h;snapshots[label]['files_sha256'][name]=h
        ww=lines(folder/'window_scores_oof.jsonl');aa=lines(folder/'answer_scores_oof.jsonl')
        assert len(ww)==12222 and len(aa)==602
        assert all(w['split']=='train' for w in ww)
        assert len({w['window_key'] for w in ww})==len(ww)
        assert len({a['item_id'] for a in aa})==len(aa)
        ts=[]
        for i,f in enumerate(folds):
            name=f'fold_{i}_calibration.json';v=read(folder/name)
            assert sha(folder/name)==fit['files_sha256'][name] and v['groups']==f
            assert v['thresholds'][METHOD]==v['selected_alpha'][METHOD]['thresholds']
            ts.append({'fold':i,'alpha':v['selected_alpha'][METHOD]['alpha'],
                       **v['thresholds'][METHOD]})
            snapshots[label]['files_sha256'][name]=sha(folder/name)
        wp=np.array([w['scores'][METHOD]>=ts[owner[w['group_id']]]['window']['threshold'] for w in ww])
        assert np.array_equal(wp,[w['predictions'][METHOD] for w in ww])
        aw=defaultdict(list)
        for w in ww:
            assert len(w['item_ids'])==1
            aw[w['item_ids'][0]].append(w)
        ans=[]
        for a in aa:
            assert a['fold']==owner[a['group_id']]
            children=aw[a['item_id']]
            assert all(w['row_id']==a['row_id'] and w['group_id']==a['group_id'] for w in children)
            value=max(w['scores'][METHOD] for w in children)
            assert value==a['scores'][METHOD];ans.append(value)
        ap=np.array([s>=ts[owner[a['group_id']]]['answer']['threshold'] for s,a in zip(ans,aa)])
        assert np.array_equal(ap,[a['predictions'][METHOD] for a in aa])
        wm=np.array([w['main_eligible'] for w in ww]);am=np.array([a['main_eligible'] for a in aa])
        assert wm.sum()==9526 and am.sum()==598
        result=read(folder/'summary.json')['methods'][METHOD]
        assert count(np.array([w['gold'] for w in ww if w['main_eligible']],int),wp[wm])==result['windows']
        assert count(np.array([a['gold'] for a in aa if a['main_eligible']],int),ap[am])==result['answers']
        refusal=np.array([a['reviewed_safe_refusal'] for a in aa],bool)
        assert refusal.sum()==117 and int(ap[refusal].sum())==result['safe_refusal_false_positives']
        assert all(a['main_eligible'] and a['gold']==0 for a in aa if a['reviewed_safe_refusal'])
        data[label]={'windows':ww,'answers':aa,'wp':wp,'ap':ap}
        thresholds[label]=ts;summary[label]=result
    for key in ('windows','answers'):
        assert [identity(r) for r in data['base'][key]]==[identity(r) for r in data['large'][key]]
    cross={}
    for unit,pkey,idkey in (('windows','wp','window_key'),('answers','ap','item_id')):
        rows=data['base'][unit];b=data['base'][pkey];l=data['large'][pkey]
        cross[unit]={}
        for gold in (1,0):
            ix=np.array([r['main_eligible'] and r['gold']==gold for r in rows],bool)
            cross[unit]['positive' if gold else 'negative']=contingency([r for r,on in zip(rows,ix) if on],b[ix],l[ix],idkey)
    aa=data['base']['answers'];rm=np.array([a['reviewed_safe_refusal'] for a in aa])
    cross['safe_refusal_false_positives']=contingency([a for a,on in zip(aa,rm) if on],data['base']['ap'][rm],data['large']['ap'][rm],'item_id')
    report={'protocol':protocol,'source_sha256':snapshots,'protocol_sha256':sha(OUT/'protocol.json'),
        'auditor_sha256':sha(__file__),'identity_and_fixed_threshold_replay_passed':True,
        'selected_per_fold':thresholds,'original_system_metrics':summary,'contingencies':cross,
        'human_gold':False,'new_fits':0,'new_thresholds':0,'oracle_F1_computed':False,
        'GPU_used':False,'public_official_test_opened':False,'old_R16_heldouts_opened_this_run':False,
        'seconds':time.perf_counter()-started}
    save(OUT/'REPORT.json',report)
    out=['# R29 / R31：固定预测的互补性','',
        '原 278 组五折身份、全部候选窗口/回答及各自既定阈值均核对通过。以下均为同一个 `r26_plus_semantic` 系统分支；未合成新预测、未选阈值、未计算 oracle F1。','',
        '| 原金标集合 | 两者报风险 | 仅 base 报风险 | 仅 large 报风险 | 两者不报风险 |',
        '|---|---:|---:|---:|---:|']
    for title,c in [('风险窗口（1063）',cross['windows']['positive']),('正常窗口（8463）',cross['windows']['negative']),
                    ('风险回答（151）',cross['answers']['positive']),('正常回答（447）',cross['answers']['negative']),
                    ('正常拒答（117，属于上行子集）',cross['safe_refusal_false_positives'])]:
        nums=[c['cells'][k]['n'] for k in ('both_risk','base_only','large_only','neither_risk')]
        out.append('| '+title+' | '+' | '.join(map(str,nums))+' |')
    pos=cross['windows']['positive']['cells'];neg=cross['windows']['negative']['cells']
    out+=['',f"base 独有检出 {pos['base_only']['n']} 个风险窗口，来自 {pos['base_only']['groups']} 个组、{pos['base_only']['answer_rows']} 条回答；同时独有误报 {neg['base_only']['n']} 个正常窗口。重叠窗口不能当独立事实样本。",'',
        '这说明两套完整系统在固定阈值下存在不同判断，不能据此证明 base 隐状态含有可额外学习的信息，或新融合必然有效：两者上游模型与训练经历、各折融合系数和阈值也不同。R16 为反复开发且 human_gold=false；本次不读取旧 heldout，不能声称它们历史上从未暴露，也不能替代公共人工 QA 结果。','']
    (OUT/'REPORT.md').write_text('\n'.join(out),'utf-8')
    compact={u:{g:{k:v['n'] for k,v in c['cells'].items()} for g,c in section.items()} for u,section in cross.items() if u in ('windows','answers')}
    compact['safe_refusals']={k:v['n'] for k,v in cross['safe_refusal_false_positives']['cells'].items()}
    print('FIXED_COMPLEMENTARITY_COMPLETE',json.dumps(compact),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        save(OUT/f'FAILURE_{time.time_ns()}.json',{'error':repr(exc),'new_fits':0,'GPU_used':False})
        raise
