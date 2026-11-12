"""Independent completion audit: recompute AUROC from saved predictions and check files."""
import json
import re
from pathlib import Path
import numpy as np
from prepare_data import ROOT,read_jsonl,sha

def pair_auc(y,s):
    a=s[y==1]; b=s[y==0]
    return float(((a[:,None]>b).sum()+.5*(a[:,None]==b).sum())/(len(a)*len(b)))

def main():
    rows=read_jsonl(ROOT/'data/processed/test.jsonl'); y=np.array([r['label'] for r in rows])
    metrics=json.loads((ROOT/'results/metrics.json').read_text())
    validated=[]
    anns={a['id']:a['spans'] for a in read_jsonl(ROOT/'data/annotations/test_spans.jsonl')}
    for m in metrics:
        path=ROOT/f'results/predictions/{m["name"]}.jsonl'
        if m['name']=='token_surprisal': continue
        pred=read_jsonl(path)
        assert [p['id'] for p in pred]==[r['id'] for r in rows]
        bags=np.array([p['bag_score'] for p in pred]); assert np.isfinite(bags).all()
        assert abs(pair_auc(y,bags)-m['response_auc'])<1e-12
        local=[]
        for r,p in zip(rows,pred):
            if p['token_scores'] is None: continue
            f=np.load(ROOT/f'data/features/{r["id"]}.npz'); scores=np.array(p['token_scores'])
            assert len(scores)==r['n_tokens'] and np.isfinite(scores).all()
            if r['label']:
                z=np.array([any(a<s['end'] and b>s['start'] for s in anns[r['id']]) for a,b in f['offsets']],dtype=int)
                if 0<z.sum()<len(z): local.append(pair_auc(z,scores))
        if local: assert abs(np.mean(local)-m['within_error_response_auc'])<1e-12
        validated.append(m['name'])
    assert len(validated)==17
    own=read_jsonl(ROOT/'data/own/reviewed.jsonl'); own_ann=read_jsonl(ROOT/'data/own/annotations.jsonl')
    assert len(own)==len(own_ann)==16
    for r,a in zip(own,own_ann):
        assert r['id']==a['id']
        for s in a['spans']+a['unresolved']: assert r['response'][s['start']:s['end']]==s['text']
    manifests=json.loads((ROOT/'results/run_manifest.json').read_text())
    for path,digest in manifests['files'].items(): assert sha(ROOT/path)==digest,path
    model=json.loads((ROOT/'results/model_manifest.json').read_text())
    assert next(x for x in model['files'] if x['file']=='model.safetensors')['upstream_lfs_verified']
    assert (ROOT/'results/comparison.png').stat().st_size>10000
    assert (ROOT/'results/token_risk_report.html').read_text(encoding='utf8').count('<article>')==89
    assert (ROOT/'results/own_generation_report.html').read_text(encoding='utf8').count('<article>')==16
    missing=[]
    for path in [ROOT/'README.md',ROOT/'results/REPORT.md']:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf8')):
            if '://' not in target and not (path.parent/target).exists(): missing.append(str(path)+':'+target)
    assert not missing,missing
    status={'status':'passed','independently_recomputed_prediction_sets':len(validated),
            'test_responses':len(rows),'own_reviewed_responses':len(own),'manifest_files_verified':len(manifests['files']),
            'report_links_valid':True,'all_test_and_own_samples_in_html':True}
    (ROOT/'results/delivery_audit.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    print(json.dumps(status,indent=2))

if __name__=='__main__': main()
