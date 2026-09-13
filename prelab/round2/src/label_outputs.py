"""Response labels and held-out local labels kept in separate artifacts."""
import argparse
import json
import re
import unicodedata
from decimal import Decimal,InvalidOperation
from collections import Counter
from common import ROOT,PRELAB,readl,writel,norm,token_f1

REFUSAL=re.compile(r'\b(unknown|unable to|cannot answer|can\x27t answer|do not know|don\x27t know|not provided|not specified|no information|not enough information)\b',re.I)

def scalar_number(text):
    s=text.lower().strip().replace(',','')
    s=re.sub(r'\s*(%|percent)$','',s); s=re.sub(r'(?<=\d)(st|nd|rd|th)$','',s)
    if re.fullmatch(r'[-+]?\d+(?:\.\d+)?',s): return Decimal(s)
    small=dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(),range(20)))
    small.update(dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(),range(20,100,10))))
    small.update(dict(zip('first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth'.split(),range(1,21))))
    words=s.replace('-',' ').split(); total=0; current=0; previous=None; seen=False
    if not words: return None
    for word in words:
        if word in small:
            value=small[word]
            if previous is not None and not (previous>=20 and previous%10==0 and 0<value<10): return None
            current+=value; previous=value; seen=True
        elif word=='and': continue
        elif word=='hundred': current=max(1,current)*100; previous=None; seen=True
        elif word in ['thousand','million']: total+=max(1,current)*({'thousand':1000,'million':1000000}[word]); current=0; previous=None; seen=True
        else: return None
    return Decimal(total+current) if seen else None

def answer_status(answer,aliases,question=''):
    def normalized(text): return norm(unicodedata.normalize('NFKC',text).replace('–','-').replace('—','-').replace('−','-'))
    a=normalized(answer); gold=[normalized(x) for x in aliases]
    if not a or REFUSAL.search(answer): return 'refusal'
    if re.search(r'how many|how much|percent|\byear|\brank|\bnumber|how long|\bdate|\bwhen',question,re.I):
        number=scalar_number(answer); numbers=[scalar_number(g) for g in aliases]
        if number is not None and number in numbers: return 'correct'
        if number is not None and all(n is not None for n in numbers): return 'candidate_error'
    if a in gold: return 'correct'
    if any(g and (' '+g+' ') in (' '+a+' ') for g in gold): return 'ambiguous'
    if max(token_f1(answer,x) for x in aliases)>0: return 'ambiguous'
    def forms(text):
        out=set(text.split())
        for word in list(out):
            for suffix in ['ing','ed','s']:
                if word.endswith(suffix) and len(word)>len(suffix)+2: out.add(word[:-len(suffix)])
        return out
    if any(forms(a)&forms(g) for g in gold): return 'ambiguous'
    return 'candidate_error'

def parse_answers(response,qas):
    try:
        start=response.index('['); values,end=json.JSONDecoder().raw_decode(response[start:])
        assert isinstance(values,list) and len(values)==3 and all(isinstance(x,str) for x in values)
        matches=list(re.finditer(r'"(?:[^"\\]|\\.)*"',response[start:start+end])); assert len(matches)==3
        return [(value,start+m.start()+1,start+m.end()-1) for value,m in zip(values,matches)]
    except (ValueError,AssertionError):
        # Accept an unambiguous three-object format only if all questions preserve order.
        questions=list(re.finditer(r'"question"\s*:\s*("(?:[^"\\]|\\.)*")',response))
        answers=list(re.finditer(r'"answer"\s*:\s*("(?:[^"\\]|\\.)*")',response))
        assert len(questions)==len(answers)==3
        assert all(norm(json.loads(m.group(1)))==norm(q['question']) for m,q in zip(questions,qas))
        return [(json.loads(m.group(1)),m.start(1)+1,m.end(1)-1) for m in answers]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task',choices=['news','trivia','rag'],required=True); ap.add_argument('--model',choices=['small','large'],required=True); args=ap.parse_args()
    run=ROOT/f'data/{args.task}_{args.model}'; rows=readl(run/'outputs.jsonl'); labeled=[]; local=[]; count=Counter()
    news_anns={a['id']:a for s in ['train','val','test'] for a in readl(PRELAB/f'data/annotations/{s}_spans.jsonl')}
    for r in rows:
        r=dict(r); ans=[]; spans=[]; ignore=[]
        if args.task=='news':
            spans=news_anns[r['id']]['spans']; status='released_human_annotation'
        elif args.task=='trivia':
            a=norm(r['response']); question=norm(r['question'])
            # Some TriviaQA aliases are the question subject itself (e.g. Jane Fonda
            # as an alias of Hanoi Jane). Merely repeating the question is not correct.
            aliases=[norm(g) for g in r['aliases'] if norm(g) and
                (norm(g)==a or (' '+norm(g)+' ') not in (' '+question+' '))]
            matched=any((' '+g+' ') in (' '+a+' ') for g in aliases)
            if r['reached_token_limit'] or REFUSAL.search(r['response']): r['label']=None; status='refusal_or_truncated'
            elif re.search(r'[.!?]\s+(?=[A-Z])',r['response']): r['label']=None; status='multiple_sentences_require_review'
            elif a in ['assassinated','yes','no','none','someone','somebody','it','they']: r['label']=None; status='nonanswer_or_ambiguous'
            elif matched and re.search(r'\b(not|isnt|wasnt|rather than)\b',a): r['label']=None; status='ambiguous_negation'
            else: r['label']=int(not matched); status='alias_match' if matched else 'alias_mismatch'
        else:
            try:
                values=parse_answers(r['response'],r['qas'])
                for i,((value,a,b),q) in enumerate(zip(values,r['qas'])):
                    state=answer_status(value,q['aliases'],q['question'])
                    item={'start':a,'end':b,'text':r['response'][a:b],'answer':value,'status':state,'question':q['question'],'aliases':q['aliases']}
                    ans.append(item)
                    if state=='candidate_error': spans.append(item)
                    elif state!='correct': ignore.append(item)
                if r['reached_token_limit'] or any(x['status']=='refusal' for x in ans): r['label']=None; status='refusal_or_truncated'
                elif spans: r['label']=1; status='at_least_one_candidate_error'
                elif ignore: r['label']=None; status='ambiguous_only'
                else: r['label']=0; status='all_exact_match'
            except (ValueError,AssertionError,json.JSONDecodeError): r['label']=None; status='format_failure'
        r['label_status']=status; count[f'{r["split"]}/{status}']+=1
        labeled.append(r); local.append({'id':r['id'],'spans':spans,'ignore':ignore,'answers':ans})
    writel(run/'labeled.jsonl',labeled); writel(run/'local_annotations.jsonl',local)
    (run/'label_manifest.json').write_text(json.dumps({'counts':dict(count),'n':len(rows),'label_provenance':'Released human spans for news; automatic reference matching for generated QA. Candidate errors need sample audit; not human labels.',
        'parser_revision':'4: RAG as revision3 plus conservative numeric equivalence for quantity/date/rank questions; Trivia unchanged from revision3',
        'trivia_scope':'Reference-target correctness, not comprehensive verification of every ancillary assertion; aliases do not establish real-world truth independently.'},indent=2),encoding='utf8')
    print(json.dumps(count,indent=2),flush=True)

if __name__=='__main__': main()
