"""Independent artifact and causal-alignment checks for this pilot."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from prepare_data import ROOT,read_jsonl,sha

def main():
    rows={s:read_jsonl(ROOT/f'data/processed/{s}.jsonl') for s in ['train','val','test']}
    groups={s:{r['group'] for r in rs} for s,rs in rows.items()}
    assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
    checks={'source_split_disjoint':True,'no_span_annotations_in_training_rows':True}
    total=0; spans=0
    for split,rs in rows.items():
        annotations={a['id']:a for a in read_jsonl(ROOT/f'data/annotations/{split}_spans.jsonl')}
        for r in rs:
            assert 'spans' not in r
            ann=annotations[r['id']]['spans']
            assert int(bool(ann))==r['label']
            for a in ann:
                assert r['response'][a['start']:a['end']].strip()==a['text'].strip()
                spans+=1
            f=np.load(ROOT/f'data/features/{r["id"]}.npz')
            assert f['hidden'].shape==(r['n_tokens'],3,896)
            assert np.isfinite(f['hidden']).all() and np.isfinite(f['nll']).all()
            assert len(f['offsets'])==len(f['hidden'])==len(f['nll'])
            assert (f['offsets'][:,0]>=0).all() and (f['offsets'][:,1]<=len(r['response'])).all()
            assert len(np.unique(f['offsets'],axis=0))>1
            total+=1
    checks.update(feature_files_valid=total,annotation_spans_valid=spans)
    # Full causal forward must match prefix-only forward at the same token.
    from extract_features import load_model,extract,LAYERS
    tok,model=load_model(); r=rows['test'][0]
    prefix=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True)
    enc=tok(prefix+r['response'],return_offsets_mapping=True,add_special_tokens=False)
    positions=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>len(prefix)]
    j=len(positions)//2; pos=positions[j]
    ids=torch.tensor([enc['input_ids'][:pos+1]],device='cuda')
    with torch.inference_mode():
        out=model.model(ids,output_hidden_states=True,use_cache=False)
    short=torch.stack([out.hidden_states[l][0,-1] for l in LAYERS]).float().cpu().numpy()
    full=np.load(ROOT/f'data/features/{r["id"]}.npz')['hidden'][j].astype(np.float32)
    relative=float(np.linalg.norm(short-full)/max(np.linalg.norm(full),1e-12))
    assert relative<.02,relative
    checks['causal_prefix_relative_error']=relative
    # Independent probability calculation checks next-token vs current-token indexing.
    with torch.inference_mode():
        logits=model.lm_head(out.last_hidden_state[0,-2]).float()
        independently_computed=-torch.log_softmax(logits,dim=-1)[ids[0,-1]].item()
    cached=float(np.load(ROOT/f'data/features/{r["id"]}.npz')['nll'][j])
    assert abs(independently_computed-cached)<.1,(independently_computed,cached)
    checks['nll_alignment_absolute_error']=abs(independently_computed-cached)
    checks['python']=sys.executable; checks['torch']=torch.__version__; checks['gpu']=torch.cuda.get_device_name()
    checks['status']='passed'
    (ROOT/'results/audit.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__': main()
