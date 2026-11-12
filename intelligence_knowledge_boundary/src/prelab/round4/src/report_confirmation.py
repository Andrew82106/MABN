"""Audit the untouched confirmation, then report successes and failures together."""
import json,pickle,hashlib,html
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
from common4 import ROOT,PRE,readl,save,sha
from prepare_confirmation import DEST
from run_confirmation import features
from fit import aggregate

NAMES={'sentences_facts':'句子＋事实点内部状态探针','sentences3':'三句直接自检分数','base_scores':'原始注意力探针'}

def main():
    protocol=json.loads((DEST/'protocol.json').read_text());manifest=json.loads((DEST/'results/manifest.json').read_text());rows=readl(DEST/'data/rows.jsonl');rr={r['id']:r for r in rows};qs=readl(DEST/'data/queries.jsonl');ann={a['id']:a['spans'] for a in readl(DEST/'data/annotations.jsonl')}
    preds=readl(DEST/'results/predictions.jsonl');ms=json.loads((DEST/'results/metrics.json').read_text());first={m['name']:m for m in json.loads((ROOT/'results/metrics.json').read_text())}
    raw={r['id']:r for r in readl(PRE/'data/raw/response.jsonl')};sources={s['source_id']:s for s in readl(PRE/'data/raw/source_info.jsonl')}
    assert sha(DEST/'protocol.json')==manifest['frozen_protocol_sha256'] and sha(DEST/'data/rows.jsonl')==protocol['rows_sha256'] and sha(DEST/'data/annotations.jsonl')==protocol['annotations_sha256']
    assert len(rows)==len(rr)==len({r['group'] for r in rows})==50 and not {r['group'] for r in rows}&{r['group'] for r in readl(ROOT/'data/rows.jsonl')}
    for r in rows:
        a=raw[r['id']];s=sources[r['source_id']];assert r['response']==a['response'] and r['evidence']==s['source_info'] and r['label']==int(bool(a['labels'])) and ann[r['id']]==a['labels']
        assert r['group']==hashlib.sha256(' '.join(r['evidence'].split()).encode()).hexdigest()
        for span in ann[r['id']]:assert span['label_type']=='Evident Conflict' and not span.get('implicit_true') and not span.get('due_to_null') and r['response'][span['start']:span['end']].strip()==span['text'].strip()
    for gen,(negative,positive) in protocol['quotas'].items():
        counts=Counter(r['label'] for r in rows if r['original_model']==gen);assert counts=={0:negative,1:positive} and positive/(positive+negative)==.4
    for q in qs:
        r=rr[q['id']];lo,hi=q['target_span'];a,b=q['sentence_span'];assert 0<=a<=lo<hi<=b<=len(r['response']) and q['target_text']==r['response'][lo:hi] and q['statement']==r['response'][a:b] and q['evidence']==r['evidence']
        assert not any(k in q for k in ['label','gold','answer','spans'])
    for p,h in protocol['frozen_dependencies'].items():assert sha(ROOT/p)==h
    b=pickle.loads((ROOT/'data/unit_inputs.pkl').read_bytes());nets=pickle.loads((ROOT/'results/checkpoints/unit_models.pkl').read_bytes());base=pickle.loads((ROOT/'results/checkpoints/base.pkl').read_bytes())['attention'];x=features(qs,DEST/'data/readouts',b['pca']);yy=np.array([rr[p['id']]['label'] for p in preds]);assert yy.sum()==20
    assert {p['id'] for p in preds}==set(rr);base_values={};offsets={};truth={}
    for r in rows:
        with np.load(r['feature_path']) as z:base_values[r['id']]=base.predict_proba(z['lookback'].astype(np.float32))[:,1].astype(np.float32);offsets[r['id']]=z['offsets'].copy()
        truth[r['id']]=np.array([int(any(lo<a['end'] and hi>a['start'] for a in ann[r['id']])) for lo,hi in offsets[r['id']]])
    subgroups=[]
    for m in ms:
        name=m['name'];c=protocol['frozen_methods'][name];assert c==m['config'];expected=[]
        if name!='base_scores':
            mid=c['model_id'];pr=x['abc'][:,1] if mid=='raw_B' else nets[mid]['model'].predict_proba(x[nets[mid]['feature']])[:,1]
        local=[]
        for p in preds:
            rid=p['id'];s=base_values[rid].copy();assert np.array_equal(p['offsets'],offsets[rid])
            if name=='base_scores':expected.append(aggregate(s,c['aggregate']))
            else:
                ids=[j for j,q in enumerate(qs) if q['id']==rid and (name=='sentences_facts' or q['kind']=='sentence')];expected.append(aggregate(pr[ids],c['aggregate']))
                for j in ids:
                    lo,hi=qs[j]['target_span'];s[(offsets[rid][:,0]<hi)&(offsets[rid][:,1]>lo)]=pr[j]
            assert np.allclose(s,p['methods'][name]['token_risks'],atol=1e-7);local.append(s)
        assert np.allclose(expected,[p['methods'][name]['score'] for p in preds],atol=1e-10)
        pp=np.asarray(expected)>=c['threshold'];assert np.array_equal(pp,[p['methods'][name]['pred'] for p in preds])
        assert abs(f1_score(yy,pp)-m['f1'])<1e-12 and abs(precision_score(yy,pp)-m['precision'])<1e-12 and abs(recall_score(yy,pp)-m['recall'])<1e-12
        assert m['tp']==int((pp&(yy==1)).sum()) and m['fp']==int((pp&(yy==0)).sum()) and m['fn']==int((~pp&(yy==1)).sum()) and m['tn']==int((~pp&(yy==0)).sum())
        assert m['token_threshold']==first[name]['token_threshold'] and abs(f1_score(np.concatenate([truth[p['id']] for p in preds]),np.concatenate(local)>=m['token_threshold'])-m['token_metrics']['f1'])<1e-12
        for gen in protocol['quotas']:
            ix=[i for i,p in enumerate(preds) if rr[p['id']]['original_model']==gen];subgroups.append({'method':name,'original_generator':gen,'n':len(ix),'errors':int(yy[ix].sum()),'f1':f1_score(yy[ix],pp[ix])})
    selected=next(m for m in ms if m['name']=='sentences_facts');assert manifest['primary_target_met']==(selected['f1']>.7)
    missed=[p for p in preds if p['label'] and not p['methods']['sentences_facts']['pred']]
    covered_misses=sum(any(q['id']==p['id'] and q['kind']=='sentence' and q['target_span'][0]<a['end'] and q['target_span'][1]>a['start'] for q in qs for a in ann[p['id']]) for p in missed)
    save(DEST/'results/error_analysis.json',{'false_negatives':len(missed),'misses_with_error_sentence_selected':covered_misses,'misses_without_error_sentence_selected':len(missed)-covered_misses,'interpretation':'Both selection coverage and the ability to judge a selected claim remain bottlenecks; post-test diagnosis, not training or selection input.'})
    save(DEST/'results/subgroups.json',subgroups);save(DEST/'results/audit.json',{'passed':True,'new_sources':50,'prior_sources_excluded':684,'original_human_labels_verified':True,'frozen_weights_and_thresholds_verified':True,'recomputed_scores_decisions_and_token_mapping':True,'primary_F1':selected['f1'],'code_sha256':{p.name:sha(p) for p in sorted((ROOT/'src').glob('*.py'))}})
    lines=['# 第四轮独立复核','',f'内部状态探针的回答级错误类F1为 **{selected["f1"]:.3f}**。'+('本次新样本点估计超过0.70。' if selected['f1']>.7 else '本次未超过0.70。'),
        '', '## 固定方法后重新测试','',
        '第一批100条新来源中，验证集预选主方法F1为0.645，未达标。另两个预设对照：三句直接自检0.709，句子＋事实点内部状态探针0.703。看到这些结果后，将第一批明确视为后续方法选择的开发信息，冻结后两种方案及原探针对照的已有权重和阈值，再收集本次50个新来源独立复核。没有用本次标签训练或调阈值，也没有用复核结果更换主方法。',
        '', '主方案固定为“句子＋事实点内部状态探针”：原探针选择最多三句；每句分别做句子核查和一个具体词语范围的核查；读取同一本地Qwen的三选一分数和四层内部状态，结合原风险等特征，用已训练的逻辑回归给每个核查单位打分，再按固定规则合并。内部状态使用原训练集拟合的16维PCA，没有在本次重拟合。',
        '', '## 新样本结果','', '| 方法 | 第一批100条F1 | 本次50条F1 | 精确率 | 召回率 | 检出/20错误 | 误报/30正常 | 逐词F1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in ms:lines.append(f'| {NAMES[m["name"]]} | {first[m["name"]]["f1"]:.3f} | {m["f1"]:.3f} | {m["precision"]:.1%} | {m["recall"]:.1%} | {m["tp"]}/20 | {m["fp"]}/30 | {m["token_metrics"]["f1"]:.3f} |')
    ci=selected['F1_95pct'];lines+=['',f'主方案固定模型的来源重采样95%区间为 [{ci[0]:.3f}, {ci[1]:.3f}]；只有50条，不能据此宣称稳定跨数据集超过0.70。逐词F1仍仅为{selected["token_metrics"]["f1"]:.3f}，回答级分类与精确定位是两回事。全部报错的回答级F1为0.571。',
        '',f'漏检的{len(missed)}条中，{len(missed)-covered_misses}条没有选到错误句子，另{covered_misses}条虽然选中错误句子，核查和探针仍未识别。后续既要改善覆盖，也要让核查包含完整事实关系。例如标注错误是“Booker had enlisted in the Army in 2014”，规则只挑出“Army”，核查一个组织名称很容易遗漏“是否入伍”的关键错误。这个例子是测试后诊断，不用于调整本次阈值或重新选模型。',
        '', '## 数据范围与限制','',
        '本次50个来源与所有此前684个来源完全隔离，20条人工明显事实冲突、30条完全无幻觉标注。只剩16个符合条件的Llama错误来源，因此在任何本次模型计算之前，将复核范围扩大到原数据中的其他生成器，按每个生成器40%错误固定抽样。它是混合生成器复核，不是第一批Llama分布的严格重复。两个准备阶段不可行的样本规模尝试均未生成冻结样本或查看预测，详见协议。',
        '', '| 原摘要生成器 | 错误 | 正常 |','|---|---:|---:|']
    for gen,(neg,pos) in protocol['quotas'].items():lines.append(f'| {gen} | {pos} | {neg} |')
    lines+=['', '生成器名称只说明RAGTruth现成摘要的来源。实验仍只运行一个本地Qwen2.5-7B NF4，没有调用GPT、其他云端模型或裁判，也没有自由生成解释。实际使用原文与现成摘要进行离线重读，未验证实时联网、自行生成摘要或中文情报领域。事实点仍是规则挑选的词语范围，不能等同于完整事实关系。',
        '', f'本次共{manifest["queries"]}次补充核查前向计算。第一批数值、失败的主选择及全部对照完整保留；不将两个阶段拼接成“完全独立测试”的总分。',
        '', '[完整指标](metrics.json) · [逐生成器结果](subgroups.json) · [全部50条样例](all_examples.html) · [冻结协议](../protocol.json) · [复算审计](audit.json) · [第一批完整报告](../../results/REPORT.md)']
    (DEST/'results/REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    parts=['<!doctype html><meta charset="utf-8"><title>独立复核全部50条新闻</title><style>body{max-width:1000px;margin:30px auto;font:16px/1.8 system-ui;padding:0 18px}article{border-top:1px solid #abc}p{white-space:pre-wrap}summary{cursor:pointer}mark{background:#ffe4cb}</style><h1>独立复核全部50条新闻</h1><p>高亮表示原人工错误范围。每条列出三个固定方法的判定，1为有事实错误，0为无错误。</p>']
    for p in sorted(preds,key=lambda x:int(x['id'])):
        r=rr[p['id']];mask=np.zeros(len(r['response']),bool)
        for a in ann[r['id']]:mask[a['start']:a['end']]=True
        text=[];i=0
        while i<len(mask):
            j=i+1
            while j<len(mask) and mask[j]==mask[i]:j+=1
            s=html.escape(r['response'][i:j]);text.append('<mark>'+s+'</mark>' if mask[i] else s);i=j
        decisions='；'.join(f'{NAMES[n]}={int(v["pred"])}' for n,v in p['methods'].items());parts.append(f'<article><h2>{r["id"]}：人工标签{r["label"]}</h2><p>{html.escape(decisions)}</p><p>{"".join(text)}</p><details><summary>新闻原文</summary><p>{html.escape(r["evidence"])}</p></details></article>')
    (DEST/'results/all_examples.html').write_text('\n'.join(parts),encoding='utf8');print('CONFIRMATION AUDIT AND REPORT PASSED',selected['f1'],flush=True)

if __name__=='__main__':main()
