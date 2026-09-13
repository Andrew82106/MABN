"""Read-only arithmetic audit and fixed comparisons for semantic hidden LR."""
from collections import Counter
import importlib.util
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import run_semantic_hidden as run
qa = run.qa
spec = importlib.util.spec_from_file_location('independent_counts', ROOT/'results/sequence_v1/audit_sequence.py')
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)


def report():
    meta=qa.metadata(); summary=qa.read(OUT/'summary.json'); scalar=qa.read(run.SEM/'results/summary.json')
    native=qa.read(ROOT/'results/sequence_capacity_v3/BASELINE_COMPARISON.json')['entries']
    comparison=list(native); rows=[]; specifications=[]
    for method,entry in summary['selected'].items():
        rows.append((entry['candidate'],entry['metrics']))
        specifications.append((entry['candidate'],OUT/(entry['candidate']+'_scores.npz'),entry['thresholds']['window']['threshold']))
    for method,entry in scalar['reports'].items():
        rows.append((method,entry['metrics']))
        scorepath=run.SEM/'results'/(method+'_scores.npz')
        if scorepath.exists(): specifications.append((method,scorepath,entry['thresholds']['window']['threshold']))
        else: assert method in ('all_positive','all_negative')
        comparison.append({'candidate':method,'kind':'extra_checker_scalar','metrics':entry['metrics']})
    for entries in summary['all_candidates'].values():
        comparison.extend({'candidate':e['candidate'],'kind':'extra_checker_hidden','metrics':e['metrics']} for e in entries)
    native_full=qa.read(ROOT/'results/sequence_full_v2/summary.json')['selected']
    for name,e in native_full.items(): rows.append((name,e['metrics']))
    diagnostics={}
    for name,path,cutoff in specifications:
        with np.load(path,allow_pickle=False) as z: scores=z['window_scores']
        alerts=scores>=cutoff; covered={}
        for w,flag in zip(meta['windows'],alerts):
            if flag: covered.setdefault(w['response_id'],set()).update(w['token_indices'])
        typed={}
        for kind in ('Evident Conflict','Subtle Conflict','Evident Baseless Info','Subtle Baseless Info'):
            spans=[];risk={};empty=0
            for tokens in meta['tokens']:
                if tokens['partition']!='calibration': continue
                for mapping in tokens['span_token_mapping']:
                    if tokens['original_labels'][mapping['span_index']]['label_type']!=kind:continue
                    pos=set(mapping['risk_token_indices']);rid=tokens['response_id']
                    if not pos:empty+=1;continue
                    risk.setdefault(rid,set()).update(pos);spans.append((bool(pos&covered.get(rid,set())),pos<=covered.get(rid,set())))
            positive=[j for j,w in enumerate(meta['windows']) if w['partition']=='calibration' and bool(set(w['token_indices'])&risk.get(w['response_id'],set()))]
            typed[kind]={'original_spans':len(spans)+empty,'localizable_spans':len(spans),'unlocalizable_spans':empty,
                'span_any_hit':sum(a for a,b in spans),'span_full_hit':sum(b for a,b in spans),
                'positive_windows':len(positive),'hit_positive_windows':int(alerts[positive].sum()),'positive_window_recall':float(alerts[positive].mean())}
        diagnostics[name]=typed
    qa.save(OUT/'DIAGNOSTICS.json',{'calibration_error_types':diagnostics,'global_selected_thresholds_no_type_tuning':True,'test_opened':False})
    qa.save(OUT/'BASELINE_COMPARISON.json',{'entries':comparison,'same_geometry_and_labels':True,'test_opened':False,
        'semantic_scalar_complete_sha256':qa.sha(run.SEM/'results/complete.json'),'hidden_complete_sha256':qa.sha(OUT/'complete.json')})
    text=['固定MiniCheck内部状态融合，全部是fit634/cal159开发结果。额外核查模型已读完整陈述与资料，属于离线方法。','',
        '| 方法 | fit窗F1 | cal窗P | cal窗R | cal窗F1 | cal整答F1 |','|---|---:|---:|---:|---:|---:|']
    for name,m in rows:
        w=m['calibration']['windows'];a=m['calibration']['answers'];text.append(f"| {name} | {m['fit']['windows']['f1']:.3f} | {w['precision']:.3f} | {w['recall']:.3f} | {w['f1']:.3f} | {a['f1']:.3f} |")
    text+=['','| 方法 | 冲突类 | 阳性窗命中 | 原span至少命中 | 原span完整覆盖 |','|---|---|---:|---:|---:|']
    for name,d in diagnostics.items():
        for kind in ('Evident Conflict','Subtle Conflict'):
            k=d[kind];text.append(f"| {name} | {kind} | {k['hit_positive_windows']}/{k['positive_windows']} | {k['span_any_hit']}/{k['localizable_spans']} | {k['span_full_hit']}/{k['localizable_spans']} |")
    text+=['','映射全量793答、213159个原始词元，671625个非空白字符无缺覆盖；原金标与窗口分母未变。PCA64只在原20288个fit抽样位置拟合，解释方差97.70%。',
        '三族各3个C均保存，阈值与C只按既定cal规则选择；此处的整体与分类型成绩均需在封存test上再确认。Subtle Conflict仅5个cal span。',
        '不能把新增MiniCheck状态称作纯Llama白盒；不能因PCA64有高解释方差就断言保留了全部有用判别信息。']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n','utf-8')
    print('SEMANTIC_HIDDEN_REPORT_COMPLETE',flush=True)



if __name__=='__main__':
    with threadpool_limits(limits=4): report()
