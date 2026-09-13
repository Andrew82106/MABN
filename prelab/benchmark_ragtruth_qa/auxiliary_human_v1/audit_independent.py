"""Read-only independent source/gold/coordinate audit; no model forward or fit.

Only official TRAIN responses from independently allowed non-QA sources are
JSON-parsed. Withheld source materials and identity fields support quarantine;
no withheld response, quality or annotation is parsed or reported.
"""
from __future__ import annotations
import ast
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import numpy as np
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
RAW = ROOT.parent / 'data/raw'
SEED = 20260912

def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def digest(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def words(s):
    return re.findall(r'\w+', s.casefold())

def leaves(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for y in x.values():
            yield from leaves(y)
    elif isinstance(x, list):
        for y in x:
            yield from leaves(y)

def same(a, b, label):
    assert a == b, (label, a, b)

def verify_sources(candidates, manifest):
    sources = {s['source_id']: s for s in rows(RAW/'source_info.jsonl')}
    ident = re.compile(r'"source_id"\s*:\s*"([^"\\]+)"')
    split_re = re.compile(r'"split"\s*:\s*"([^"\\]+)"')
    splits = {}
    identity_lines = 0
    with (RAW/'response.jsonl').open(encoding='utf-8') as f:
        for line in f:
            sid, split = ident.search(line)[1], split_re.search(line)[1]
            assert split in ('train', 'test')
            assert splits.setdefault(sid, split) == split
            identity_lines += 1
    qa = {r['source_id']: r for r in rows(ROOT/'data/source_index.jsonl')}
    blocked = {s for s in splits if splits[s] == 'test'} | {s for s in qa if qa[s]['partition'] != 'fit'}
    # Build an adjacency graph and flood-fill components, independently of the
    # producer's union-find. Every edge has a real shared material/entity witness.
    adj = {s: set() for s in sources}
    grams, entities = {}, {}
    edge_types = defaultdict(set)
    blank_entities = []
    def edge(a, b, why):
        if a != b:
            adj[a].add(b); adj[b].add(a)
            edge_types[tuple(sorted((a,b)))].add(why)
    for sid, s in sources.items():
        info = s['source_info']
        parts = (re.split(r'(?:^|\n\s*\n)passage\s+\d+:', info['passages'], flags=re.I)
                 if s['task_type'] == 'QA' else leaves(info))
        for part in parts:
            w = words(part)
            for i in range(len(w)-19):
                key = digest(' '.join(w[i:i+20]))
                other = grams.setdefault(key, sid)
                edge(sid, other, 'shared_consecutive20_words')
        if s['task_type'] == 'Data2txt':
            key = tuple(tuple(words(str(info.get(k,'')))) for k in ('name','address','city','state'))
            if not any(key):
                blank_entities.append(sid)
            edge(sid, entities.setdefault(key,sid), 'same_business_name_address_city_state')
    group = {}; components = {}
    for sid in sorted(sources):
        if sid in group:
            continue
        members = {sid}; queue = [sid]
        for current in queue:
            for n in adj[current]:
                if n not in members:
                    members.add(n); queue.append(n)
        g = 'rtaux_group_' + digest('|'.join(sorted(members)))[:16]
        components[g] = members
        group.update({m:g for m in members})
    eligible = {s for s in sources if splits[s]=='train' and sources[s]['task_type'] in ('Summary','Data2txt')}
    quarantine = {s for s in eligible if components[group[s]] & blocked}
    allowed = eligible - quarantine
    actual_quarantine = list(rows(OUT/'quarantined_source_index.jsonl'))
    same({r['source_id'] for r in actual_quarantine}, quarantine, 'quarantine identities')
    for r in actual_quarantine:
        same(r['group_id'], group[r['source_id']], 'quarantine group')
    actual_edges = {(r['source_a'],r['source_b']):set(r['reasons']) for r in rows(OUT/'overlap_edge_index.jsonl')}
    same(actual_edges, dict(edge_types), 'material/entity edges')
    witnesses = []
    for sid in sorted(quarantine):
        q=deque([sid]); previous={sid:None}; end=None
        while q:
            n=q.popleft()
            if n in blocked:
                end=n; break
            for near in sorted(adj[n]):
                if near not in previous:
                    previous[near]=n; q.append(near)
        assert end is not None
        path=[]; n=end
        while n is not None:
            path.append(n); n=previous[n]
        witnesses.append({'quarantined_source':sid, 'path':path[::-1],
                          'blocking_identity_reason':'official_test_source' if splits[end]=='test' else 'QA_nonfit_source'})
    by_id={r['response_id']:r for r in candidates}
    assert len(by_id)==len(candidates)
    parsed=Counter(); matched=set(); bad=[]; raw_good=[]; raw_keys=set(); duplicate_ids=[]
    with (RAW/'response.jsonl').open(encoding='utf-8') as f:
        for line in f:
            sid=ident.search(line)[1]
            if sid not in allowed or split_re.search(line)[1]!='train':
                continue
            # Guard BEFORE parsing the JSON: this audit never parses test gold.
            raw=json.loads(line); source=sources[sid]
            assert raw['split']=='train' and raw['source_id']==sid
            parsed[source['task_type']]+=1
            if raw['quality']!='good':
                bad.append(raw['id']); continue
            key=json.dumps([sid,raw['response'],raw['labels']],sort_keys=True,ensure_ascii=False)
            if key in raw_keys:
                duplicate_ids.append(raw['id']); continue
            raw_keys.add(key); raw_good.append(raw['id'])
            r=by_id[raw['id']]; matched.add(raw['id'])
            for left,right in [('original_response','response'),('labels','labels'),('model','model'),('quality','quality'),('temperature','temperature')]:
                same(r[left],raw[right],f'raw unchanged {raw["id"]}:{left}')
            same(r['source_id'],sid,'source identity')
            same(r['group_id'],group[sid],'component identity')
            assert r['official_split']=='train' and r['partition']=='auxiliary_candidate_fit'
            assert r['new_labels_generated'] is False and r['currently_used_in_training'] is False
            same(r['released_prompt'],source['prompt'],'released prompt')
            same(r['task_type'],source['task_type'],'task')
            same(r['source_dataset'],source['source'],'source dataset')
            a,b=r['evidence_original_prompt_range']
            same(r['released_prompt'][a:b],r['retrieved_passages'],'original evidence rendering')
            if r['task_type']=='Summary':
                same(r['retrieved_passages'],source['source_info'],'summary material')
                same(r['question'],source['prompt'][:a].rstrip(),'summary instruction')
                same(source['prompt'][b:],'\noutput:','summary removed suffix')
            else:
                same(ast.literal_eval(r['retrieved_passages']),source['source_info'],'all structured fields')
                same(r['question'],source['prompt'][:a].rstrip(),'data2txt instruction')
                same(source['prompt'][b:],'\nOverview:','data2txt removed suffix')
            same(digest(r['original_response']),r['answer_sha256'],'answer digest')
            same(digest(r['released_prompt']),r['prompt_sha256'],'prompt digest')
            for label in r['labels']:
                assert type(label['start']) is int and type(label['end']) is int
                assert 0<=label['start']<label['end']<=len(r['original_response'])
                same(r['original_response'][label['start']:label['end']],label['text'],'original span alignment')
    same(matched,set(by_id),'complete retained answer coverage')
    same(set(bad),{r['response_id'] for r in rows(OUT/'quality_excluded_index.jsonl')},'quality exclusions')
    same(set(duplicate_ids),{r['response_id'] for r in rows(OUT/'duplicate_index.jsonl')},'duplicate exclusions')
    counts={}
    for task in ('Summary','Data2txt'):
        sub=[r for r in candidates if r['task_type']==task]
        counts[task]={'answers':len(sub),'sources':len({r['source_id'] for r in sub}),
                     'groups':len({r['group_id'] for r in sub}),'risk_answers':sum(bool(r['labels']) for r in sub),
                     'span_types':dict(Counter(l['label_type'] for r in sub for l in r['labels']))}
    same(counts,manifest['counts'],'all manifest task counts')
    totals={'candidate_train_sources_before_quarantine':len(eligible),'quarantined_train_sources':len(quarantine),
            'unique_sources':len({r['source_id'] for r in candidates}),'unique_groups':len({r['group_id'] for r in candidates}),
            'answers':len(candidates),'quality_excluded_rows':len(bad),'span_alignment_issues':0,
            'same_source_exact_answer_gold_duplicates_removed':len(duplicate_ids),
            'parsed_train_rows_by_task':dict(parsed)}
    for k,v in totals.items(): same(v,manifest[k],k)
    retained={r['source_id'] for r in candidates}
    same(retained,allowed,'all retained eligible sources')
    fitlinked=sorted(s for s in retained if any(q in qa and qa[q]['partition']=='fit' for q in components[group[s]]))
    return {'status':'passed','counts':counts,**totals,'identity_only_lines':identity_lines,
            'test_answer_or_gold_JSON_parsed':False,'blank_Data2txt_entity_keys':blank_entities,
            'retained_sources_material_linked_to_QA_fit':fitlinked,'quarantine_witnesses':witnesses,
            'task_reassembly':'All released evidence and task instructions preserved; only final output:/Overview: cues omitted.'}

def verify_tokens(candidates, report):
    chosen=set(random.Random(SEED).sample(range(len(candidates)),100))
    sample_ids=[]
    llama=AutoTokenizer.from_pretrained(ROOT.parent/'models/Llama-2-7b-chat-hf',local_files_only=True)
    bert=AutoTokenizer.from_pretrained(ROOT.parent/'models/ModernBERT-base',local_files_only=True)
    stats={t:Counter() for t in ('Summary','Data2txt')}; lengths=defaultdict(list)
    maxerr=0.; checked=0; boundary_crossing=0; nspans=0
    stream=rows(OUT/'token_inputs.jsonl')
    for i,r in enumerate(candidates):
        z=next(stream); checked+=1
        for key in ('response_id','source_id','group_id','task_type','answer_sha256'):
            same(z[key],r[key],f'token {key}')
        text=r['original_response']; off=z['response_token_offsets']; rawoff=z['response_token_offsets_raw']
        assert len(off)==len(rawoff)==len(z['response_token_ids'])
        same(off,[[max(0,a),min(len(text),b)] for a,b in rawoff],'clipped raw coordinates')
        boundary_crossing+=sum(a<0 or b>len(text) for a,b in rawoff)
        marked=set()
        for lab in r['labels']:
            marked.update(j for j in range(lab['start'],lab['end']) if text[j].isalnum())
        nspans+=len(r['labels'])
        lexical=[];risk=[]
        for a,b in off:
            assert 0<=a<=b<=len(text)
            lexical.append(any(c.isalnum() for c in text[a:b]))
            risk.append(any(c in marked for c in range(a,b)))
        same(lexical,z['lexical_mask'],'full lexical oracle')
        same(risk,z['risk_mask'],'full human character risk oracle')
        same(int(bool(r['labels'])),z['answer_risk'],'answer label')
        assert bool(any(risk))==bool(r['labels'])
        begin=z['answer_encoder_start'];end=z['answer_encoder_end']
        assert len(begin)==len(end)==len(z['input_ids'])
        owners=[[] for _ in text]
        for j,(a,b) in enumerate(zip(begin,end)):
            if a==-1 or b==-1:
                assert a==b==-1;continue
            assert 0<=a<=b<=len(text)
            for c in range(a,b):
                if not text[c].isspace():owners[c].append(j)
        assert all(owners[c] for c in range(len(text)) if not text[c].isspace())
        # Reconstruct every mapping coefficient from raw answer characters;
        # no producer helper is imported.
        rr=[];cc=[];ww=[]
        for k,(a,b) in enumerate(off):
            chars=[c for c in range(a,b) if not text[c].isspace()]
            acc=defaultdict(float)
            for c in chars:
                for owner in owners[c]:acc[owner]+=1/(len(chars)*len(owners[c]))
            for owner,val in sorted(acc.items()):
                rr.append(k);cc.append(owner);ww.append(val)
        same(rr,z['mapping'][0],'all mapping row indices')
        same(cc,z['mapping'][1],'all mapping encoder indices')
        error=float(np.max(np.abs(np.asarray(ww,dtype=np.float32)-np.asarray(z['mapping'][2])))) if ww else 0.
        maxerr=max(maxerr,error); assert error<1e-7
        sums=np.bincount(rr,weights=z['mapping'][2],minlength=len(off))
        for k,(a,b) in enumerate(off):
            expect=any(not c.isspace() for c in text[a:b])
            assert abs(sums[k]-int(expect))<2e-7
        if i in chosen:
            sample_ids.append({'index':i,'response_id':r['response_id']})
            prefix='<s>[INST] '+r['released_prompt']+' [/INST] '
            whole=prefix+text
            enc=llama(whole,add_special_tokens=False,padding=False,truncation=False,return_offsets_mapping=True)
            ix=[j for j,(a,b) in enumerate(enc['offset_mapping']) if b>len(prefix) and a<len(whole) and b>a]
            same([enc['input_ids'][j] for j in ix],z['response_token_ids'],'sample exact Llama token ids')
            same([[enc['offset_mapping'][j][0]-len(prefix),enc['offset_mapping'][j][1]-len(prefix)] for j in ix],rawoff,'sample exact Llama raw coordinates')
            prefix=r['retrieved_passages']+bert.sep_token+r['question']+bert.sep_token
            whole=prefix+text
            enc=bert(whole,add_special_tokens=True,truncation=False,return_offsets_mapping=True)
            same(enc['input_ids'],z['input_ids'],'sample full ModernBERT ids')
            reconstructed=[(max(0,a-len(prefix)),min(len(text),b-len(prefix))) if b>len(prefix) and a<len(whole) else (-1,-1) for a,b in enc['offset_mapping']]
            same([a for a,b in reconstructed],begin,'sample exact encoder start')
            same([b for a,b in reconstructed],end,'sample exact encoder end')
        assert len(z['input_ids'])<=bert.model_max_length
        windows=[(s,min(s+4,len(off))) for s in range(max(1,len(off)-3))]
        eligible=[(a,b) for a,b in windows if any(lexical[a:b])]
        stats[r['task_type']].update(answers=1,encoder_input_tokens=len(begin),raw_answer_tokens=len(off),
             lexical_answer_tokens=sum(lexical),risk_answer_tokens=sum(risk),windows=len(eligible),
             risk_windows=sum(any(risk[a:b]) for a,b in eligible))
        lengths[r['task_type']].append(len(begin))
        if checked%2000==0:print('INDEPENDENT TOKEN AUDIT',checked,flush=True)
    assert next(stream,None) is None
    for t in stats:
        stats[t]['max_input_tokens']=max(lengths[t]);stats[t]['median_input_tokens']=float(np.median(lengths[t]))
    actual={k:dict(v) for k,v in stats.items()}
    same(actual,report['stats'],'all token/window/length totals')
    same(checked,report['answers'],'token coverage')
    same(report['exceptions'],[],'producer exceptions')
    return {'status':'passed','answers_with_all_gold_and_all_mapping_coefficients_rebuilt':checked,
            'original_human_spans_checked':nspans,'max_mapping_float32_difference':maxerr,
            'tokenization_replay_sample_size':len(chosen),'sample_seed':SEED,'sample':sample_ids,
            'boundary_crossing_tokens_retained':boundary_crossing,'stats':actual,
            'coverage_limit':'Exact tokenizer replay checks 100 seeded answers; original spans, character gold, every mapping coefficient and every 4-BPE window count check all 9678 answers.',
            'native_trace_claim':False,'no_truncation_in_replayed_inputs':True}

def main():
    started=time.time()
    manifest=read(OUT/'manifest.json'); tokenreport=read(OUT/'token_input_preparation.json')
    candidates=list(rows(OUT/'candidate_fit.jsonl'))
    codepaths=[ROOT/'src/prepare_auxiliary_human.py',ROOT/'src/prepare_auxiliary_tokens.py']
    locked={str(p):sha(p) for p in codepaths+[OUT/'manifest.json',OUT/'token_input_preparation.json']}
    for name,expected in manifest['artifacts_sha256'].items():
        same(sha(OUT/name),expected,'staged artifact '+name)
    same(sha(OUT/'token_inputs.jsonl'),tokenreport['token_inputs_sha256'],'token data snapshot')
    result={'status':'running','scope':'Independent read-only auxiliary human TRAIN data audit; no model forward, GPU, fit or official test answer/gold access.',
            'script_sha256':sha(__file__),'code_and_manifest_sha256':locked}
    try:
        result['source_and_labels']=verify_sources(candidates,manifest)
        print('INDEPENDENT SOURCE AUDIT passed',flush=True)
        result['token_coordinates']=verify_tokens(candidates,tokenreport)
        for p,v in locked.items():same(sha(p),v,'unchanged during audit '+p)
        result['status']='passed'
        result['conclusion']='Suitable staged human-supervised auxiliary tasks for later QA detection training, with original QA calibration/test unchanged. It is not new QA data and has not been used to train any model by this preparation/audit.'
        result['limitations']=[
            'Exact 20-word material and full business-identity grouping does not prove exhaustive semantic/entity/event independence.',
            'Source identities, not 9678 generator answers, count independent material; 1614 retained sources form 1558 conservative material groups.',
            'Summary/Data2txt differ from QA: original task instructions and schema fields are retained; labels are not reinterpreted as world-false-only.',
            '4 raw BPE windows retain punctuation and use OR over lexical risk. Their roughly 1.97 million overlapping windows must not be reported as independent observations.',
            'No mixture weights, fitted model, calibration selection or improvement evidence is produced here; future auxiliary/QA mixing and source balancing must be predefined.'
        ]
    except Exception as e:
        result['status']='failed';result['failure']=repr(e)
        raise
    finally:
        result['wall_seconds']=time.time()-started
        (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'answers':len(candidates),'sources':manifest['unique_sources'],
                      'groups':manifest['unique_groups'],'quarantine':manifest['quarantined_train_sources'],
                      'wall_seconds':result['wall_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
