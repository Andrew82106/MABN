"""Source-only QA grouping; no response/label file is opened."""
from pathlib import Path
import json,re,unicodedata,hashlib,math,itertools
from collections import defaultdict,Counter

ROOT=Path(__file__).resolve().parents[1];PRE=ROOT.parent
OUT=ROOT/'data';OUT.mkdir(parents=True,exist_ok=True)
def readl(p):
    with p.open(encoding='utf-8') as f:
        for s in f:
            if s.strip():yield json.loads(s)
def norm(s):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold()))
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def save(name,r): (OUT/name).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n','utf-8')
src={r['source_id']:r for r in readl(PRE/'data/raw/source_info.jsonl')}
qa={sid:r for sid,r in src.items() if r['task_type']=='QA'}
prior=list(readl(PRE/'round20_localization_optimization/research/ragtruth_qa_metadata_index.jsonl'))
meta={r['source_id']:r for r in prior}

# Material that has already been used, including early news and QA inputs.
oldpaths=[PRE/f'data/processed/{s}.jsonl' for s in ('train','val','test')]
oldpaths +=[PRE/'round2/data/news_confirmation/rows.jsonl',PRE/'round3/data/rows.jsonl',PRE/'round4/data/rows.jsonl',PRE/'round4/confirmation/data/rows.jsonl']
news_ids=set()
for p in oldpaths:
    news_ids.update(r['source_id'] for r in readl(p))
docs={};questions={}
for sid in news_ids:
    assert src[sid]['task_type']=='Summary'
    docs['old_news:'+str(sid)]=[src[sid]['source_info']]
for rn in ['round6_evidence_grounding','round7_evidence_grounding','round9_evidence_binding','round10_dual_granularity','round16_dataset_expansion']:
    for fname in ['inputs.jsonl','dev_inputs.jsonl']:
        p=PRE/rn/'data'/fname
        if not p.exists():continue
        for r in readl(p):
            key='old_'+rn+':'+r['row_id']
            docs[key]=[v['text'] for v in r['passages']]
            questions[key]=r.get('questions',[r.get('question','')])
for fname in ['trivia_questions.jsonl','rag_questions.jsonl']:
    for r in readl(PRE/'round2/data'/fname):
        key='old_round2:'+r['id']
        questions[key]=[r['question']] if 'question' in r else [x['question'] for x in r['qas']]
        if 'evidence' in r:docs[key]=[r['evidence']] if isinstance(r['evidence'],str) else [str(r['evidence'])]
qa_key={sid:'qa:'+str(sid) for sid in qa}
for sid,r in qa.items():
    parts=re.split(r'(?:^|\n\s*\n)passage\s+\d+:',r['source_info']['passages'],flags=re.I)
    parts=[p.strip() for p in parts if p.strip()]
    assert len(parts)==3,(sid,len(parts))
    docs[qa_key[sid]]=parts
    questions[qa_key[sid]]=[r['source_info']['question']]

# Exact paragraph or 20 consecutive normalized words: conservative material links.
paragraphs=defaultdict(set);windows=defaultdict(set)
for key,parts in docs.items():
    for p in parts:
        ws=norm(p).split()
        if len(ws)>=8:paragraphs[h(' '.join(ws))].add(key)
        for j in range(max(0,len(ws)-19)):
            windows[h(' '.join(ws[j:j+20]))].add(key)
pair_evidence=defaultdict(lambda:{'exact_passage_hashes':[],'shared_20word_window_count':0,'example_window_hash':None})
for ph,keys in paragraphs.items():
    qkeys=[x for x in keys if x.startswith('qa:')]
    for a in qkeys:
        for b in keys:
            if a==b:continue
            pair=tuple(sorted([a,b]))
            if ph not in pair_evidence[pair]['exact_passage_hashes']:pair_evidence[pair]['exact_passage_hashes'].append(ph)
for wh,keys in windows.items():
    qkeys=[x for x in keys if x.startswith('qa:')]
    seen=set()
    for a in qkeys:
        for b in keys:
            if a==b:continue
            pair=tuple(sorted([a,b]))
            if pair in seen:continue
            seen.add(pair);pair_evidence[pair]['shared_20word_window_count']+=1
            pair_evidence[pair]['example_window_hash']=wh
material=[{'a':a,'b':b,**v} for (a,b),v in sorted(pair_evidence.items())]
save('material_overlap_audit.json',{'rule':'same >=8-word entire passage or any exact normalized 20-word source window; no answer/labels used','edges':material})

# Question-only semantic candidate links. These are reviewed before grouping.
stop=set('a an the of in on at to and or for from with by is are was were be been being do does did how what when where why who which whom whose can could should would will may might it its you your i my they their we our has have had that this these those as if mean means meaning called definition define much many number cost first last one two three used use us new most about into over after before'.split())
qt={sid:set(norm(r['source_info']['question']).split())-stop for sid,r in qa.items()}
df=Counter(w for ws in qt.values() for w in ws)
semantic=[]
ids=sorted(qa,key=str)
for i,a in enumerate(ids):
    for b in ids[i+1:]:
        common=qt[a]&qt[b]
        if not common:continue
        rare=[w for w in common if len(w)>=5 and df[w]<=5]
        jac=len(common)/max(1,len(qt[a]|qt[b]))
        if rare or (len(common)>=2 and jac>=.4):
            semantic.append({'a':a,'b':b,'question_a':qa[a]['source_info']['question'],'question_b':qa[b]['source_info']['question'],
             'shared_tokens':sorted(common),'rare_tokens':sorted(rare),'jaccard':jac,'decision':'pending_source_question_review'})
save('question_link_candidates.json',{'method':'question lexical candidates only; false positive and false negative links possible','pairs':semantic})
save('source_audit_summary.json',{'qa_sources':len(qa),'old_news_sources':len(news_ids),'old_source_documents_or_input_rows':sum(not k.startswith('qa:') for k in docs),
 'qa_qa_material_edges':sum(e['a'].startswith('qa:') and e['b'].startswith('qa:') for e in material),
 'qa_old_material_edges':sum(not (e['a'].startswith('qa:') and e['b'].startswith('qa:')) for e in material),
 'question_candidate_pairs':len(semantic),'source_info_sha256':hashlib.sha256((PRE/'data/raw/source_info.jsonl').read_bytes()).hexdigest(),
 'limitations':['MS MARCO release lacks original source URLs per retrieved passage.','Question linking and exact text checks do not certify all entity aliases or specific-event relationships.','Current source review is not independent researcher human annotation.'],
 'new_response_or_label_file_opened':False})
print(json.dumps(json.loads((OUT/'source_audit_summary.json').read_text('utf-8')),ensure_ascii=False,indent=2))
