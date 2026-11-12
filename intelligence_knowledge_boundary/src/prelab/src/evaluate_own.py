"""Apply frozen probes to independently generated target-model responses."""
import json
import numpy as np
import torch
from prepare_data import ROOT,read_jsonl,write_jsonl
from evaluate import predict,metrics,make_html,token_labels

def span(text,quote,reason):
    assert text.count(quote)==1,(quote,text.count(quote))
    start=text.index(quote)
    return {'start':start,'end':start+len(quote),'text':quote,'meta':reason,'label_type':'Assistant-reviewed explicit conflict'}

def main():
    review=json.loads((ROOT/'data/own/review_decisions.json').read_text(encoding='utf8'))
    rows=read_jsonl(ROOT/'data/own/generated.jsonl')
    assert set(review['decisions'])=={r['id'] for r in rows}
    anns=[]; feats=[]
    for r in rows:
        d=review['decisions'][r['id']]; r['label']=d['label']
        anns.append({'id':r['id'],'spans':[span(r['response'],q,d['reason']) for q in d['conflicts']],
                     'unresolved':[span(r['response'],q,d['reason']) for q in d['unresolved']],
                     'label':d['label'],'reason':d['reason']})
        feats.append(dict(np.load(ROOT/f'data/own/features/{r["id"]}.npz')))
    write_jsonl(ROOT/'data/own/reviewed.jsonl',rows); write_jsonl(ROOT/'data/own/annotations.jsonl',anns)
    selected=json.loads((ROOT/'results/selection.json').read_text())['display_checkpoint']
    ck,bags,tokens=predict(ROOT/f'results/checkpoints/{selected}.pt',rows,feats)
    valid=[i for i,r in enumerate(rows) if r['label'] is not None]
    vf=[]; vt=[]
    for i in valid:
        # Mask unsupported/ambiguous positions: they are neither verified errors nor verified correct facts.
        keep=token_labels(feats[i]['offsets'],anns[i]['unresolved'])==0
        vf.append({'offsets':feats[i]['offsets'][keep]}); vt.append(tokens[i][keep])
    y=np.array([rows[i]['label'] for i in valid]); va=[anns[i] for i in valid]
    m=metrics(y,bags[valid],vt,vf,va,ck['threshold'])
    result={'n_generated':len(rows),'n_labeled':len(valid),'n_explicit_conflict':int(y.sum()),
            'n_no_detected_error':int((y==0).sum()),'n_unresolved':len(rows)-len(valid),
            'checkpoint':selected,'annotation_provenance':review['provenance'],
            'limitation':'Only one negative response: classification AUROC/F1 are not stable estimates. These are descriptive transfer diagnostics, not confirmatory results.',
            'metrics_descriptive_only':m}
    (ROOT/'results/own_generation_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    write_jsonl(ROOT/'results/predictions/own_generation.jsonl',[{'id':r['id'],'label':r['label'],'bag_score':float(b),'token_scores':s.tolist()} for r,b,s in zip(rows,bags,tokens)])
    make_html(rows,feats,anns,bags,tokens,ROOT/'results/own_generation_report.html','Qwen 自然生成补充审阅 · 单执行助手标注，仅供探索')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
