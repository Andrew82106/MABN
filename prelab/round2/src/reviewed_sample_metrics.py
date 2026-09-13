"""Secondary fixed audit sample; never used for model or parameter selection."""
import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from common import ROOT,readl

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='large'); ap.add_argument('--task',default='trivia'); args=ap.parse_args()
    assert args.model=='large' and args.task=='trivia'
    dest=ROOT/'results/trivia_large'; review=json.loads((ROOT/'results/trivia_large_label_review.json').read_text())
    bundle=readl(ROOT/'data/trivia_large/review_bundle.jsonl'); preds={r['id']:r for r in readl(dest/'predictions.jsonl')}
    kept=[]
    for r in bundle:
        label=review['exceptions'].get(r['id'],{}).get('reviewed_label',r['label'])
        if label is not None and r['id'] in preds: kept.append((r['id'],label))
    y=np.array([label for _,label in kept]); metrics=[]
    for name in preds[kept[0][0]]['predictions']:
        p=np.array([preds[rid]['predictions'][name]['bag'] for rid,_ in kept])
        metrics.append({'name':name,'reviewed_reference_target_auc':float(roc_auc_score(y,p))})
    out={'n':len(kept),'errors':int(y.sum()),'selection':'Fixed seed481 reference/answer audit performed before 7B Trivia probe fitting',
        'scope':'Small stratified auxiliary sample; neither model selection nor a substitute for full factuality gold labels',
        'metrics':metrics,'reviewed_ids':[rid for rid,_ in kept]}
    (dest/'reviewed_sample_metrics.json').write_text(json.dumps(out,indent=2),encoding='utf8'); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
