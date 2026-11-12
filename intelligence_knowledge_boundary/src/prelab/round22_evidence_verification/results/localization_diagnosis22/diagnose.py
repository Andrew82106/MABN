"""Post-hoc frozen R22 error decomposition; no fitting or cutoff changes."""
from pathlib import Path
from collections import defaultdict, Counter
import json, re, hashlib

OUT=Path(__file__).resolve().parent
RESULT=OUT.parent
MAIN='verifier_learned_slots';BASE='slots_base'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')


def confusion(rows,name):
    yy=[r for r in rows if r['main_eligible']];tp=fp=fn=tn=0
    for r in yy:
        y,p=r['gold'],r['predictions'][name]
        if y:tp+=int(p);fn+=int(not p)
        else:fp+=int(p);tn+=int(not p)
    return {'n':len(yy),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else None,
      'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def rate(n,d):return {'count':n,'denominator':d,'fraction':n/d if d else None}


def run():
    complete=read(RESULT/'complete22.json')
    for n,h in complete['files_sha256'].items():assert sha(RESULT/n)==h
    original_hash=sha(RESULT/'complete22.json')
    windows=lines(RESULT/'window_scores_oof22.jsonl');answers=lines(RESULT/'answer_scores_oof22.jsonl')
    summary=read(RESULT/'summary22.json');lengths=read(RESULT/'lengths22.json')
    raw_index={r['item_id']:r for r in lines(RESULT/'answer_index22.jsonl')}
    index={r['item_id']:dict(r,start=raw_index[r['item_id']]['start'],end=raw_index[r['item_id']]['end']) for r in answers};byitem=defaultdict(list)
    for w in windows:byitem[w['item_ids'][0]].append(w)
    eligible=[w for w in windows if w['main_eligible']]
    riskbounds={}
    for iid,ww in byitem.items():
        rr=[w for w in ww if w['main_eligible'] and w['gold']==1]
        # Bounds of positive windows, deliberately not falsely called span bounds.
        if rr:riskbounds[iid]=(min(w['start'] for w in rr),max(w['end'] for w in rr))
    def item_class(w):
        a=index[w['item_ids'][0]]
        return 'risk_answer' if a['gold']==1 else 'safe_refusal' if a['reviewed_safe_refusal'] else 'supported_answer' if a['gold']==0 else 'unresolved'
    def answer_correct(w):
        a=index[w['item_ids'][0]];return 'answer_decision_correct' if a['predictions'][MAIN]==bool(a['gold']) else 'answer_decision_wrong'
    def length_bin(w):
        n=lengths[w['row_id']];return '<=16' if n<=16 else '17..32' if n<=32 else '>32'
    def position(w):
        a=index[w['item_ids'][0]];mid=(w['start']+w['end'])/2
        z=(mid-a['start'])/max(a['end']-a['start'],1)
        return 'first_third' if z<1/3 else 'middle_third' if z<2/3 else 'last_third'
    def sentence(w):
        a=index[w['item_ids'][0]];boundaries=[m.end() for m in re.finditer(r'[.!?]+\s+(?=[A-Z0-9"\u201c])',a['text'])]
        local=(w['start']+w['end'])/2-a['start'];j=sum(b<=local for b in boundaries)
        return 'only_sentence' if not boundaries else 'first_sentence' if j==0 else 'last_sentence' if j==len(boundaries) else 'middle_sentence'
    def risk_location(w):
        iid=w['item_ids'][0]
        if iid not in riskbounds:return 'no_risk_window_in_answer'
        lo,hi=riskbounds[iid]
        return 'before_positive_window_extent' if w['end']<=lo else 'after_positive_window_extent' if w['start']>=hi else 'inside_or_overlapping_positive_window_extent'
    changes=defaultdict(list)
    for w in eligible:
        bp=w['predictions'][BASE];mp=w['predictions'][MAIN];y=w['gold']
        kind=('lost_TP_new_FN' if y else 'removed_FP') if bp and not mp else ('gained_TP' if y else 'new_FP') if mp and not bp else 'unchanged'
        changes[kind].append(w)
    fields={'true_answer_label':item_class,'answer_decision':answer_correct,'answer_length':length_bin,'window_position_thirds':position,
      'sentence_position_heuristic':sentence,'relation_to_positive_window_extent':risk_location,'condition':lambda w:w['condition'],'category':lambda w:w['category']}
    transitions={}
    for kind,ww in changes.items():
        transitions[kind]={'windows':len(ww),'answers':len({w['item_ids'][0] for w in ww}),
          'strata':{key:dict(Counter(fn(w) for w in ww)) for key,fn in fields.items()},
          'window_keys':[w['window_key'] for w in ww]}
    all_by_field={}
    for field,fn in fields.items():
        categories=sorted({fn(w) for w in eligible})
        all_by_field[field]={value:{name:confusion([w for w in eligible if fn(w)==value],name) for name in (BASE,MAIN)} for value in categories}
    aa=[]
    for a in answers:
        if not a['main_eligible']:continue
        ww=[w for w in byitem[a['item_id']] if w['main_eligible']]
        counts={name:confusion(ww,name) for name in (BASE,MAIN)}
        peaks=[w for w in ww if w['scores'][MAIN]==max(x['scores'][MAIN] for x in ww)] if ww else []
        aa.append({'item_id':a['item_id'],'row_id':a['row_id'],'fold':a['fold'],'gold':a['gold'],
          'answer_pred_main':a['predictions'][MAIN],'answer_pred_slots':a['predictions'][BASE],
          'safe_refusal':a['reviewed_safe_refusal'],'bpe_length':lengths[a['row_id']],
          'window_counts':counts,'all_peak_windows_risky':bool(peaks) and all(w['gold']==1 for w in peaks),
          'category':a['category'],'condition':a['condition']})
    risk=[a for a in aa if a['gold']==1];tp=[a for a in risk if a['answer_pred_main']];fn=[a for a in risk if not a['answer_pred_main']]
    correct=[a for a in aa if bool(a['gold'])==a['answer_pred_main']]
    asserted_correct=[a for a in correct if a['window_counts'][MAIN]['n']]
    analysis={'all_risk_answers':len(risk),'correctly_flagged_risk_answers':len(tp),'missed_risk_answers':len(fn),
      'among_correctly_flagged_risk_answers':{
        'zero_risk_window_hit':rate(sum(a['window_counts'][MAIN]['tp']==0 for a in tp),len(tp)),
        'some_risk_window_hit':rate(sum(a['window_counts'][MAIN]['tp']>0 for a in tp),len(tp)),
        'any_risk_window_missed':rate(sum(a['window_counts'][MAIN]['fn']>0 for a in tp),len(tp)),
        'any_nonrisk_window_false_alarm':rate(sum(a['window_counts'][MAIN]['fp']>0 for a in tp),len(tp)),
        'any_window_error':rate(sum(a['window_counts'][MAIN]['fp']+a['window_counts'][MAIN]['fn']>0 for a in tp),len(tp)),
        'all_peak_windows_overlap_labeled_risk':rate(sum(a['all_peak_windows_risky'] for a in tp),len(tp))},
      'among_all_correct_answer_decisions':{'answers':len(correct),'asserted_with_local_gold':len(asserted_correct),
        'any_window_error_in_asserted':rate(sum(a['window_counts'][MAIN]['fp']+a['window_counts'][MAIN]['fn']>0 for a in asserted_correct),len(asserted_correct))},
      'among_missed_risk_answers':{'some_risk_window_hit':rate(sum(a['window_counts'][MAIN]['tp']>0 for a in fn),len(fn)),
        'all_peak_windows_overlap_labeled_risk':rate(sum(a['all_peak_windows_risky'] for a in fn),len(fn))}}
    totals={unit:{name:confusion(rr,name) for name in (BASE,MAIN)} for unit,rr in [('windows',windows),('answers',answers)]}
    for unit,ms in totals.items():
        for name,counts in ms.items():
            for k in ('tp','fp','fn','tn','f1','precision','recall'):assert counts[k]==summary['methods'][name][unit][k]
    assert len(changes['lost_TP_new_FN'])-len(changes['gained_TP'])==totals['windows'][BASE]['tp']-totals['windows'][MAIN]['tp']
    assert len(changes['new_FP'])-len(changes['removed_FP'])==totals['windows'][MAIN]['fp']-totals['windows'][BASE]['fp']
    examples={}
    for kind in ('new_FP','lost_TP_new_FN','gained_TP','removed_FP'):
        seen=set();chosen=[]
        for w in sorted(changes[kind],key=lambda w:w['window_key']):
            iid=w['item_ids'][0]
            if iid in seen:continue
            seen.add(iid);a=index[iid]
            chosen.append({'window_key':w['window_key'],'response':a['text'],'window':w['text'],'gold_window':w['gold'],
              'answer_gold':a['gold'],'answer_prediction':a['predictions'][MAIN],
              'window_score_slots':w['scores'][BASE],'window_score_main':w['scores'][MAIN],
              'selection':'First three distinct answers by window_key; illustration only, not random error sample'})
            if len(chosen)==3:break
        examples[kind]=chosen
    result={'status':'passed','scope':'Post-hoc descriptive analysis of already frozen R22 train-source CV; no fit, no cutoff change, no oracle metric',
      'complete22_sha256':original_hash,'summary22_sha256':sha(RESULT/'summary22.json'),'totals':totals,'transitions':transitions,
      'all_window_counts_by_strata':all_by_field,'localization_conditioned_on_answer_decision':analysis,'per_answer':aa,'illustrative_examples':examples,
      'limitations':['Sentence positions use punctuation heuristic, not a gold syntactic annotation.',
        'Positive-window extent includes four-BPE boundary overlap; it is not the exact factual-error span.',
        'Any window error is a strict all-windows diagnostic, not the same as no useful localization.',
        'Main score and slots have identical within-answer ranking except numeric ties; this analysis cannot establish causality.',
        'Risk answer gold means unsupported/contradicted by supplied evidence, not necessarily false in the world.']}
    save(OUT/'diagnosis.json',result)
    report=['这是冻结结果的事后拆解，没有训练、改阈值或挑选新方法。','',
      '| 改动 | 窗口数 | 涉及回答数 |','|---|---:|---:|']
    for k in ('new_FP','removed_FP','lost_TP_new_FN','gained_TP'):report.append(f"| {k} | {transitions[k]['windows']} | {transitions[k]['answers']} |")
    report+=['',f"联合模型正确识别了{len(tp)}/{len(risk)}份风险回答。以下比例都以这{len(tp)}份回答为分母："]
    text={'zero_risk_window_hit':'一个风险窗口也没命中','some_risk_window_hit':'至少命中一个风险窗口','any_risk_window_missed':'仍漏掉部分风险窗口','any_nonrisk_window_false_alarm':'仍把无风险窗口标红','any_window_error':'至少存在一个窗口误报或漏报'}
    for k,label in text.items():
        q=analysis['among_correctly_flagged_risk_answers'][k];report.append(f"- {label}：{q['count']}/{q['denominator']}。")
    report+=['','新增FP与漏检的真整答标签、位置和长度分布：']
    for k in ('new_FP','lost_TP_new_FN'):
        report.append(f"- {k}："+'；'.join(f"{field}={transitions[k]['strata'][field]}" for field in ('true_answer_label','answer_decision','window_position_thirds','answer_length','sentence_position_heuristic')))
    report+=['','“整答核查正确”不等于“所有位置均正确”。上面的任何窗口错误比例很严格，应和至少命中率一起读；标红4BPE与金标重叠，也不等于已完整定位到一个事实错误。位置/长度统计仅描述关联，不证明错误由这些因素造成。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n','utf-8')
    assert original_hash==sha(RESULT/'complete22.json')
    for n,h in complete['files_sha256'].items():assert sha(RESULT/n)==h
    print(json.dumps({'totals':totals,'changes':{k:v['windows'] for k,v in transitions.items()},'answer_localization':analysis},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
