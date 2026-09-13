"""Independent frozen-dataset counts; imports no production alignment/annotation code."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def readj(p):
    return json.loads(p.read_text(encoding='utf-8'))


def readl(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_canonical_item(g, a):
    """A failed parser's null boundaries must never mean the whole response."""
    assert len(g['items']) == 1 and a['item_id'] == g['row_id'] + '__1'
    item = g['items'][0]
    for key in ('item_id', 'text', 'start', 'end'):
        assert a[key] == item[key], (g['row_id'], key)
    if item['parse_ok']:
        assert isinstance(a['start'], int) and isinstance(a['end'], int)
        assert 0 <= a['start'] <= a['end'] <= len(g['response'])
        assert a['text'] == g['response'][a['start']:a['end']]
    else:
        assert a['text'] == '' and a['start'] is None and a['end'] is None
        assert a['localization_status'] == 'unresolved'
        assert a['original_risk'] is None and not a['risk_spans']
        assert a['answer_gold'] is None and not a['reviewed_safe_refusal']


def raw_gold(g, a):
    text = g['response']
    resolved = a['original_stance'] == 'asserted' and a['original_risk'] in (0, 1) and a['localization_status'] == 'resolved'
    known = {i for i in range(a['start'], a['end']) if text[i].isalnum()} if a['start'] is not None else set()
    bad = {i for s in a['risk_spans'] for i in range(s['start'],s['end']) if text[i].isalnum()}
    tokens = []
    for n, (left,right) in enumerate(g['response_token_offsets']):
        assert 0 <= left <= right <= len(text)
        chars = {i for i in range(left,right) if text[i].isalnum()}
        eligible = bool(chars) and chars <= known and resolved
        tokens.append({'token_key':g['row_id']+f'__token{n}', 'token_index':n,
                       'token_id':g['response_token_ids'][n], 'start':left,'end':right,
                       'text':text[left:right], 'lexical':bool(chars), 'main_eligible':eligible,
                       'gold':int(bool(chars & bad)) if eligible else None})
    windows = []
    if resolved:
        indices = [i for i,(l,r) in enumerate(g['response_token_offsets']) if r>a['start'] and l<a['end']]
        assert indices and indices == list(range(indices[0],indices[-1]+1))
        for k in range(max(1,len(indices)-3)):
            raw = indices[k:k+4]
            eligible = [tokens[i] for i in raw if tokens[i]['main_eligible']]
            if not eligible:
                continue
            l,r = max(a['start'],tokens[raw[0]]['start']),min(a['end'],tokens[raw[-1]]['end'])
            windows.append({'window_key':a['item_id']+f'__w4__start{raw[0]}',
                            'width':4,'actual_width':len(raw),'short_window':len(raw)<4,
                            'raw_token_indices':raw,'token_keys':[t['token_key'] for t in eligible],
                            'start':l,'end':r,'text':text[l:r],
                            'gold':int(any(t['gold']==1 for t in eligible))})
    return tokens,windows


def run():
    freeze_path = ROOT/'data/annotation_freeze.json'
    assert freeze_path.exists(), 'Run only after all labels have been frozen'
    freeze = readj(freeze_path)
    assert freeze['status'] == 'assistant_annotated_independently_reviewed_frozen'
    for name,digest in freeze['files_sha256'].items():
        assert sha(ROOT/name)==digest, name
    inputs_freeze = readj(ROOT/'data/input_freeze.json')
    assert sha(ROOT/'data/input_freeze.json') == freeze['input_freeze_sha256']
    for name,digest in inputs_freeze['files_sha256'].items():
        assert sha(ROOT/name)==digest,name
    legacy = readj(ROOT/'data/legacy_files_snapshot.json')
    for name,digest in legacy['files_sha256'].items():
        assert sha(ROOT.parent/name)==digest,name
    manifest = readj(ROOT/'data/generation_manifest.json')
    assert sha(ROOT/'data/generation_manifest.json')==freeze['generation_manifest_sha256']
    assert manifest['complete'] and manifest['generated']==800
    inputs = readl(ROOT/'data/inputs.jsonl')
    assert len(inputs)==800 and len({r['row_id'] for r in inputs})==800
    groups = defaultdict(set)
    summary = {}
    for split in ('train','validation','test'):
        selected = [r for r in inputs if r['split']==split]
        labels = readl(ROOT/'data'/f'annotations_{split}.jsonl')
        actual_tokens = readl(ROOT/'data'/f'tokens_{split}.jsonl')
        actual_windows = readl(ROOT/'data'/f'windows_k4_{split}.jsonl')
        ready = readl(ROOT/'data'/f'dataset_{split}.jsonl')
        aa = {a['row_id']:a for a in labels}
        assert len(aa)==len(labels)==len(selected)==len(ready)
        assert set(aa)=={r['row_id'] for r in selected}
        ready = {r['input']['row_id']:r for r in ready}
        tt = {t['token_key']:t for t in actual_tokens}
        ww = {w['window_key']:w for w in actual_windows}
        assert len(tt)==len(actual_tokens) and len(ww)==len(actual_windows)
        expected_t, expected_w = [], []
        safe_ids = set(readj(ROOT/'data'/f'safe_refusals_{split}.json')['safe_refusal_item_ids'])
        assert safe_ids=={a['item_id'] for a in labels if a['reviewed_safe_refusal']}
        for row in selected:
            rid=row['row_id'];a=aa[rid];p=ROOT/'data/generation_records'/(rid+'.json');g=readj(p)
            assert sha(p)==manifest['record_sha256'][rid]==a['source_generation_sha256']
            verify_canonical_item(g,a)
            assert a['human_gold'] is False and a['token_scores_viewed'] is False
            assert a['initial_annotator'].split(':')[0] != a['independent_reviewer'].split(':')[0]
            spans=a['risk_spans']
            assert bool(spans)==(a['original_risk']==1)
            for s in spans:
                assert a['start']<=s['start']<s['end']<=a['end'] and g['response'][s['start']:s['end']]==s['text']
            assert all(x['end']<=y['start'] for x,y in zip(spans,spans[1:]))
            if a['original_stance']!='asserted' or a['localization_status']=='unresolved':
                assert a['original_risk'] is None and not spans
            if a['reviewed_safe_refusal']:
                assert a['original_stance']=='abstained' and a['localization_status']=='excluded' and a['answer_gold']==0
            else:
                assert a['answer_gold']==a['original_risk']
            assert ready[rid]['input']==row and ready[rid]['response']==g['response'] and ready[rid]['annotation']==a
            for k in ('response_token_ids','response_token_offsets'):
                assert ready[rid][k]==g[k]
            assert len(g['response_token_ids'])==len(g['response_token_offsets'])
            tokens,windows=raw_gold(g,a);expected_t.extend(tokens);expected_w.extend(windows)
            if not g['items'][0]['parse_ok']:
                assert not windows and all(not t['main_eligible'] and t['gold'] is None for t in tokens)
            for token in tokens:
                assert all(tt[token['token_key']][k]==v for k,v in token.items()), token['token_key']
            for window in windows:
                assert all(ww[window['window_key']][k]==v for k,v in window.items()),window['window_key']
            groups[row['group_id']].add(split)
        assert len(expected_t)==len(tt) and len(expected_w)==len(ww)
        summary[split]={'questions':len({r['question_id'] for r in selected}),
                        'event_or_subject_groups':len({r['group_id'] for r in selected}),'answers':len(labels),
                        'supported_answers':sum(a['original_risk']==0 for a in labels),
                        'risky_answers':sum(a['original_risk']==1 for a in labels),
                        'safe_refusals':len(safe_ids),'unresolved_or_other':sum(a['answer_gold'] is None for a in labels),
                        'localization_answers':sum(a['localization_status']=='resolved' for a in labels),
                        'eligible_tokens':sum(t['main_eligible'] for t in expected_t),
                        'risk_tokens':sum(t['gold']==1 for t in expected_t),
                        'windows_4':len(expected_w),'risk_windows_4':sum(w['gold'] for w in expected_w),
                        'question_categories':dict(Counter(r['category'] for r in selected if r['condition']=='complete'))}
    assert all(len(s)==1 for s in groups.values())
    assert summary==readj(ROOT/'results/dataset_statistics.json')['new']
    result={'status':'passed','annotation_freeze_sha256':sha(freeze_path),'auditor_sha256':sha(Path(__file__)),
            'new':summary,'checks':['frozen source/output/label hashes','legacy bytes unchanged','800 exact answer spans',
             'safe refusal answer-only0; no unknown token0','original BPE axes including punctuation','sliding4 OR risk labels',
             'whole question groups remain one split','independent counts equal exported statistics'],
            'limitation':'Checks consistency and counts, not annotation semantic truth. No probe training or performance scoring.'}
    (ROOT/'results/INDEPENDENT_DATASET_AUDIT16.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','answers':800,'splits':summary},ensure_ascii=False))


if __name__=='__main__':
    run()
