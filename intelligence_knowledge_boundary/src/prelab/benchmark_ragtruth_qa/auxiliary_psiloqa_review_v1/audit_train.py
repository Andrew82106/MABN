"""Read only PsiloQA TRAIN plus existing RAGTruth material-only identities.

No text repair, model, tokenizer, training, validation/test data access or export
of a new training split. All English rows, including failures, remain inventoried.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import re
import time
from urllib.parse import urlsplit, unquote

import numpy as np
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
RAW = ROOT.parent / 'data/raw/source_info.jsonl'
FAVA = ROOT / 'auxiliary_fava_v2'
RX = re.compile(r'\[HAL\]|\[/HAL\]')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):
    with Path(p).open(encoding='utf-8') as f:
        for s in f:
            if s.strip(): yield json.loads(s)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def canon(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
def save(name, x): (OUT/name).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def savel(name, records):
    with (OUT/name).open('w', encoding='utf-8') as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False)+'\n')


def parse_markup(s):
    chunks = []; spans = []; start = None; length = 0; previous = 0; errors = []
    for m in RX.finditer(s):
        text = s[previous:m.start()]; chunks.append(text); length += len(text)
        if m.group() == '[HAL]':
            if start is not None: errors.append('nested_open')
            start = length
        else:
            if start is None: errors.append('orphan_close')
            else:
                spans.append([start, length])
                if start == length: errors.append('empty_span')
                start = None
        previous = m.end()
    chunks.append(s[previous:])
    if start is not None: errors.append('unclosed_open')
    return ''.join(chunks), spans, errors


def summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a): return {'n': 0}
    return {'n': len(a), 'sum': float(a.sum()), 'mean': float(a.mean()),
            **{k: float(v) for k, v in zip(['min','p25','p50','p75','p90','p95','p99','max'],
                np.percentile(a,[0,25,50,75,90,95,99,100]))}}


def span_stats(texts):
    sizes = [len(re.findall(r'\w+', s)) for s in texts]
    bins = Counter('0' if n==0 else '1' if n==1 else '2-3' if n<=3 else '4-9' if n<=9 else '10-19' if n<=19 else '20+' for n in sizes)
    predicates = {
        'contains_unicode_digit': lambda s: any(c.isdigit() for c in s),
        'only_ASCII_digits_after_strip': lambda s: re.fullmatch(r'[0-9]+',s.strip()) is not None,
        'numeric_notation_after_strip': lambda s: re.fullmatch(r'[0-9][0-9,.\s/%:+–—-]*',s.strip()) is not None,
    }
    return {'characters': summary([len(s) for s in texts]), 'unicode_word_runs': summary(sizes),
            'word_run_bins': dict(sorted(bins.items())),
            'string_predicates': {k: {'count': sum(fn(s) for s in texts),
                'fraction': sum(fn(s) for s in texts)/len(texts) if texts else None} for k,fn in predicates.items()},
            'not_semantic_error_categories': True}


class Union:
    def __init__(self, n): self.p=list(range(n))
    def find(self, i):
        while self.p[i]!=i: self.p[i]=self.p[self.p[i]]; i=self.p[i]
        return i
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)


def article_key(url):
    u=urlsplit(url)
    return u.netloc.lower()+unquote(u.path).replace('_',' ')


def run():
    assert not (OUT/'complete.json').exists(), 'Do not overwrite completed review'
    tick=time.perf_counter(); download=read(OUT/'download_manifest.json')
    for f in download['files']: assert sha(OUT/f['path'])==f['sha256']
    upstream=read(FAVA/'manifest.json')
    for n in ('rt_source_identity_index.jsonl','rt_material_group_index.jsonl'):
        assert sha(FAVA/n)==upstream['artifacts_sha256'][n]
    rulepath=ROOT/'src/prepare_auxiliary_human.py'
    assert sha(RAW)==upstream['sources_sha256'][str(RAW.resolve())]
    assert sha(rulepath)==upstream['sources_sha256'][str(rulepath.resolve())]
    spec=importlib.util.spec_from_file_location('psilo_readonly_material_rules',rulepath)
    rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
    protocol={
        'scope':'Only pinned PsiloQA train parquet; inspect every lang=en row. Language code is metadata, not independently verified text language.',
        'alignment':'Strict no-nested HAL scanner; Unicode Python character end-exclusive offsets. Original llm_answer and labels unchanged. Strict pass requires markup projection exactly original answer, valid ordered nonoverlapping labels, exact ordered span pairs. Strip comparison is diagnostic only, no transformed training examples.',
        'inventory':'All rows retained. Empty labels are not called clean when HAL positive or alignment fails. Exact article URL (decoded path, underscores→spaces, host lowercase) and exact nonempty passage hash connected inside English train. No other split inspected.',
        'numeric_diagnostics':'Unicode isdigit any; pure ASCII digits after strip; numeric notation characters [0-9,. whitespace / % : + dash]. These are string statistics, not dates, quantities or semantic error taxonomy.',
        'overlap':'Reused frozen FAVA source-only material identities and original source_info evidence only. Exact nonempty raw part SHA or 20 consecutive Unicode word runs after casefold. Return unique material-group matches, preserve source witness. Propagate protected matches over English article/material groups for reported quarantine only.',
        'prohibited':'No validation/test files or answers, no models/GPU/tokenizer/training, no edits to source text/labels or existing data. No external judge. No claim of human gold, split independence or full paraphrase/entity isolation.',
        'preliminary_read':'A preliminary full English alignment count preceded this frozen audit; no model or selection used it.',
    }
    source_paths=[Path(__file__),OUT/'download_manifest.json',RAW,rulepath,FAVA/'manifest.json',FAVA/'rt_source_identity_index.jsonl',FAVA/'rt_material_group_index.jsonl']
    save('AUDIT_PROTOCOL.json',protocol)
    save('audit_freeze.json',{'source_sha256':{str(p.resolve()):sha(p) for p in source_paths},'protocol_sha256':sha(OUT/'AUDIT_PROTOCOL.json')})
    table=pq.read_table(OUT/'train-00000-of-00001.parquet')
    allrows=table.to_pylist(); lang=Counter(r['lang'] for r in allrows)
    en=[(i,r) for i,r in enumerate(allrows) if r['lang']=='en']
    data=[r for _,r in en]; assert len(data)==16115 and len(allrows)==63792
    checks=[];failures=[]; spans_ledger=[]; counts=Counter(); offsets=Counter(); questions=Counter();
    lengths=defaultdict(list); label_texts=[]; exact_texts=[]; tag_texts=[]
    article=Union(len(data)); seen_url={};seen_hash={}; materials=defaultdict(list); urlmembers=defaultdict(list)
    identical=defaultdict(list);sameinput=defaultdict(list)
    for i,(rowindex,r) in enumerate(en):
        answer=r['llm_answer']; clean,marked,errors=parse_markup(r['annotated_span']); labs=r['labels']
        for key in ('id','lang','wiki_title','wiki_url','llm_checkpoint','wiki_passage','question','golden_answer','llm_answer','annotated_span','complexity'):
            if not isinstance(r[key],str) or not r[key].strip():errors.append('empty_or_nonstring_'+key)
        valid=isinstance(labs,list) and all(isinstance(x,list) and len(x)==2 and all(isinstance(v,int) for v in x) and 0<=x[0]<x[1]<=len(answer) for x in labs)
        ordered=valid and all(a[1]<=b[0] for a,b in zip(labs,labs[1:]))
        exact=clean==answer; same=marked==labs; passed=not errors and valid and ordered and exact and same
        fields={'schema_and_tags_valid':not errors,'labels_in_bounds':valid,'labels_ordered_nonoverlap':ordered,
                'projection_exact':exact,'projection_strip_exact_diagnostic_only':clean.strip()==answer.strip(),
                'labels_equal_markup_positions':same,'strict_exact_pass':passed,'labels_positive':bool(labs),'HAL_positive':bool(marked),
                'empty_labels_but_HAL_positive':not labs and bool(marked)}
        counts.update(k for k,v in fields.items() if v)
        checked={'train_row_index':rowindex,'english_index':i,'id':r['id'],**fields,'errors':errors,
            'answer_sha256':digest(answer),'wiki_passage_sha256':digest(r['wiki_passage']),
            'original_labels':labs,'markup_projection_spans':marked,'markup_projection_sha256':digest(clean)}
        checks.append(checked)
        if not passed: failures.append({**checked,'original_train_record':r})
        if len(labs)==len(marked):
            offsets.update((a-c,b-d) for (a,b),(c,d) in zip(labs,marked))
        for j,(a,b) in enumerate(labs if valid else []):
            s=answer[a:b];label_texts.append(s)
            if passed:exact_texts.append(s)
            spans_ledger.append({'id':r['id'],'span_index':j,'start':a,'end':b,'text':s,'strict_row_pass':passed,
                'characters':len(s),'word_runs':len(re.findall(r'\w+',s)),'contains_digit':any(c.isdigit() for c in s),
                'pure_ASCII_digits':re.fullmatch(r'[0-9]+',s.strip()) is not None})
        tag_texts.extend(clean[a:b] for a,b in marked)
        words=re.findall(r'\w+',r['question'].casefold());questions[words[0] if words else '<empty>']+=1
        for key in ('question','wiki_passage','golden_answer','llm_answer'):
            lengths[key+'_characters'].append(len(r[key]));lengths[key+'_word_runs'].append(len(re.findall(r'\w+',r[key])))
        h=digest(r['wiki_passage']);u=article_key(r['wiki_url']);materials[h].append(i);urlmembers[u].append(i)
        if r['wiki_passage'].strip():article.join(i,seen_hash.setdefault(h,i))
        if u:article.join(i,seen_url.setdefault(u,i))
        identical[digest(canon(r))].append(i)
        sameinput[digest(canon([r['wiki_passage'],r['question'],answer]))].append(i)
    groups=defaultdict(list)
    for i in range(len(data)):groups[article.find(i)].append(i)
    group_ids={i:'psilo_article_'+digest('|'.join(sorted(data[j]['id'] for j in members))) for members in groups.values() for i in members}
    savel('english_rows.jsonl',checks);savel('alignment_failures.jsonl',failures);savel('label_spans.jsonl',spans_ledger)
    savel('english_article_groups.jsonl',[{'group_id':group_ids[m[0]],'ids':[data[i]['id'] for i in m],
        'wiki_urls':sorted(set(data[i]['wiki_url'] for i in m)),'material_sha256':sorted(set(digest(data[i]['wiki_passage']) for i in m))} for m in groups.values()])
    duplicates=[]
    for key,m in sameinput.items():
        if len(m)>1: duplicates.append({'input_key_sha256':key,'ids':[data[i]['id'] for i in m],
            'different_label_lists':len({canon(data[i]['labels']) for i in m})>1,
            'different_annotated_span':len({data[i]['annotated_span'] for i in m})>1})
    savel('same_material_question_answer_duplicates.jsonl',duplicates)
    print('ENGLISH_ALIGNMENT',len(data),'strict',counts['strict_exact_pass'],'failures',len(failures),'groups',len(groups),flush=True)
    # Only the material-only source file and the already built source identities.
    identities={r['source_id']:r for r in lines(FAVA/'rt_source_identity_index.jsonl')}
    sources={r['source_id']:r for r in lines(RAW)};assert set(identities)==set(sources) and len(sources)==2965
    exact_index=defaultdict(list);grams={}
    for sid,source in sources.items():
        for pi,part in enumerate(rules.evidence_parts(source)):
            if part.strip():exact_index[digest(part)].append((sid,pi))
            words=rules.norm(part)
            for pos in range(len(words)-19):
                h=digest(' '.join(words[pos:pos+20])); old=grams.setdefault(h,(sid,pi,pos))
                assert identities[old[0]]['material_group_id']==identities[sid]['material_group_id']
    matches=[]
    for h,members in materials.items():
        text=data[members[0]]['wiki_passage'];found={}
        for sid,pi in exact_index.get(h,[]):
            ident=identities[sid];g=ident['material_group_id']
            found[g]={'reason':'exact_evidence_SHA','rt_source_witness':sid,'rt_part_index':pi,'rt_identity':ident,'shared_20gram_count':0}
        words=rules.norm(text)
        for pos in range(len(words)-19):
            h20=digest(' '.join(words[pos:pos+20]));hit=grams.get(h20)
            if hit is None:continue
            sid,pi,rpos=hit;ident=identities[sid];g=ident['material_group_id']
            found.setdefault(g,{'reason':'consecutive20_normalized_words','rt_source_witness':sid,'rt_part_index':pi,
                'rt_word_start':rpos,'psilo_word_start':pos,'shared20_sha256':h20,'shared20_text':' '.join(words[pos:pos+20]),
                'rt_identity':ident,'shared_20gram_count':0})['shared_20gram_count']+=1
        for g,match in found.items():matches.append({'wiki_passage_sha256':h,'psilo_ids':[data[i]['id'] for i in members],
            'psilo_wiki_urls':sorted(set(data[i]['wiki_url'] for i in members)),**match})
    savel('source_material_matches.jsonl',matches)
    direct={rid for m in matches if m['rt_identity']['blocked_after_existing_edge_propagation'] for rid in m['psilo_ids']}
    blocked_groups={group_ids[i] for i,r in enumerate(data) if r['id'] in direct}
    quarantine=[{'id':r['id'],'group_id':group_ids[i],'direct_material_match':r['id'] in direct,
                 'strict_exact_alignment':checks[i]['strict_exact_pass'],'report_only_not_removed':True}
                for i,r in enumerate(data) if group_ids[i] in blocked_groups]
    savel('reported_quarantine.jsonl',quarantine)
    strict=[r for r,c in zip(data,checks) if c['strict_exact_pass']]
    result={
        'status':'completed_readonly_inventory_not_training_ready_certification','all_official_train_rows':len(allrows),
        'language_metadata_counts':dict(sorted(lang.items())),'english_metadata_rows':len(data),'schema':str(table.schema),
        'alignment':dict(counts),'alignment_failure_rows':len(failures),'label_spans':len(label_texts),'HAL_spans':len(tag_texts),
        'label_markup_delta_counts_same_number_of_spans':[{ 'start_delta':k[0],'end_delta':k[1],'n':v} for k,v in offsets.most_common()],
        'raw_label_answer_counts':{'positive':sum(bool(r['labels']) for r in data),'empty':sum(not r['labels'] for r in data)},
        'strict_aligned_silver_answer_counts':{'positive':sum(bool(r['labels']) for r in strict),'unmarked':sum(not r['labels'] for r in strict)},
        'generator_counts':dict(Counter(r['llm_checkpoint'] for r in data).most_common()),
        'complexity_counts':dict(Counter(r['complexity'] for r in data).most_common()),
        'question_first_word_counts':dict(questions.most_common()),
        'question_prefix_counts':{x:sum(bool(re.match(x,r['question'],re.I)) for r in data) for x in ['^when\\b','^how many\\b','^how much\\b','^in (what|which) year\\b','^what year\\b','^what\\b','^who\\b','^where\\b']},
        'lengths':{k:summary(v) for k,v in lengths.items()},
        'span_statistics_all_released_labels_including_mismatch':span_stats(label_texts),
        'span_statistics_strict_exact_rows_only':span_stats(exact_texts),
        'span_statistics_markup_projection_diagnostic_only':span_stats(tag_texts),
        'duplicates':{'unique_ids':len({r['id'] for r in data}),'raw_passages':len(materials),'article_urls':len(urlmembers),
            'article_or_exact_material_groups':len(groups),'group_size':summary([len(m) for m in groups.values()]),
            'unique_questions':len({r['question'] for r in data}),
            'identical_full_records_excess':sum(len(m)-1 for m in identical.values()),
            'same_material_question_answer_groups':len(duplicates),'same_input_excess':sum(len(d['ids'])-1 for d in duplicates),
            'same_input_different_label_lists_groups':sum(d['different_label_lists'] for d in duplicates)},
        'source_overlap':{'RAGTruth_sources':len(sources),'protected_RT_identities':sum(r['blocked_after_existing_edge_propagation'] for r in identities.values()),
            'unique_RT_20grams':len(grams),'deduplicated_psilo_material_RT_group_pairs':len(matches),
            'direct_match_answer_ids':len({rid for m in matches for rid in m['psilo_ids']}),
            'direct_protected_match_answers':len(direct),'reported_quarantine_article_groups':len(blocked_groups),'reported_quarantine_answers':len(quarantine),
            'strict_aligned_answers_after_reported_quarantine':sum(c['strict_exact_pass'] and group_ids[i] not in blocked_groups for i,c in enumerate(checks)),
            'no_rows_removed_or_added_to_training':True,'paraphrase_or_complete_entity_independence_proved':False},
        'validation_test_files_read':False,'models_downloaded':False,'GPU_used':False,'trained':False,
        'existing_data_modified':False,'human_gold':False,'elapsed_seconds':time.perf_counter()-tick,
    }
    save('summary.json',result)
    produced=['AUDIT_PROTOCOL.json','audit_freeze.json','english_rows.jsonl','alignment_failures.jsonl','label_spans.jsonl',
        'english_article_groups.jsonl','same_material_question_answer_duplicates.jsonl','source_material_matches.jsonl','reported_quarantine.jsonl','summary.json']
    save('complete.json',{'status':'complete','files_sha256':{n:sha(OUT/n) for n in produced},'download_manifest_sha256':sha(OUT/'download_manifest.json'),
                          'official_train_only':True,'trained':False,'GPU_used':False})
    print(json.dumps({'alignment':dict(counts),'overlap':result['source_overlap'],'seconds':result['elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':run()
