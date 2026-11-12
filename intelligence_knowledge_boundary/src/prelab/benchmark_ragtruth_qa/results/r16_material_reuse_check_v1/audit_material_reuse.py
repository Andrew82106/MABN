"""Finite 20-word material-overlap check of explicitly opened training files.

No score/model/threshold use. Never open R16 validation/test or QA test.
"""
from collections import defaultdict, Counter
import ast
import hashlib
import json
from pathlib import Path
import re
import time

OUT=Path(__file__).resolve().parent
QA=OUT.parents[1]
PRELAB=QA.parent
INPUTS={
    'R16_train':PRELAB/'round16_dataset_expansion/data/dataset_train.jsonl',
    'QA_fit':QA/'fit_expansion/data/fit.jsonl',
    'auxiliary_human_fit_candidates':QA/'auxiliary_human_v1/candidate_fit.jsonl'}
WIDTH=20

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()

def digest(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()

def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def parts(task,text):
    if task=='QA':
        # No invented adjacency between separate retrieved passages.
        return [p for p in re.split(r'(?:^|\n\s*\n)passage\s+\d+:',text,flags=re.I) if p.strip()]
    # For structured data, published rendering is the actual evidence string.
    # Include it literally, as requested, and label its task in any match.
    return [text]

def grams(text):
    folded=text.casefold()
    positions=(list(range(len(text))) if len(folded)==len(text)
               else [i for i,c in enumerate(text) for _ in c.casefold()])
    matches=list(re.finditer(r'\w+',folded))
    words=[m.group() for m in matches]
    for i in range(len(words)-WIDTH+1):
        normalized=' '.join(words[i:i+WIDTH])
        yield digest(normalized),positions[matches[i].start()],positions[matches[i+WIDTH-1].end()-1]+1,normalized

def main():
    start=time.perf_counter();snap={k:{'path':str(p.resolve()),'sha256':sha(p)} for k,p in INPUTS.items()}
    documents={};rowids=set();questions=set();groups=set();conditions=Counter();occurrences=0
    for wrapper in rows(INPUTS['R16_train']):
        r=wrapper['input'];assert r['split']=='train'
        assert r['row_id'] not in rowids;rowids.add(r['row_id'])
        questions.add(r['question_id']);groups.add(r['group_id']);conditions[r['condition']]+=1
        for passage in r['passages']:
            body=passage['text'];h=digest(body);occurrences+=1
            d=documents.setdefault(h,{'text':body,'titles':set(),'occurrences':[]})
            d['titles'].add(passage['title'])
            d['occurrences'].append({'row_id':r['row_id'],'question_id':r['question_id'],
                'group_id':r['group_id'],'condition':r['condition'],'visible_title':passage['title']})
    assert (len(rowids),len(questions),len(groups))==(602,301,278)
    target=defaultdict(list);short=0
    for h,d in documents.items():
        seen=set()
        for key,a,b,normalized in grams(d['text']):
            if key in seen:continue
            target[key].append((h,a,b));seen.add(key)
        short+=not bool(seen)
    corpora={};matches=[]
    for corpus in ('QA_fit','auxiliary_human_fit_candidates'):
        sources={};answer_count=0
        for r in rows(INPUTS[corpus]):
            assert r['official_split']=='train'
            assert r['partition']==('fit' if corpus=='QA_fit' else 'auxiliary_candidate_fit')
            answer_count+=1
            task='QA' if corpus=='QA_fit' else r['task_type']
            assert task in ('QA','Summary','Data2txt')
            s=sources.setdefault(r['source_id'],{'text':r['retrieved_passages'],'group_id':r['group_id'],
                                               'task':task,'response_ids':[]})
            assert s['text']==r['retrieved_passages'] and s['group_id']==r['group_id'] and s['task']==task
            s['response_ids'].append(r['response_id'])
        assert answer_count==(3680 if corpus=='QA_fit' else 9678)
        pairs={};body_pair_count=set();evidence_parts=0;unique_piece_hashes=set()
        for sid,s in sources.items():
            for part_index,body in enumerate(parts(s['task'],s['text'])):
                evidence_parts+=1;unique_piece_hashes.add(digest(body))
                for key,other_a,other_b,normalized in grams(body):
                    for h,original_a,original_b in target.get(key,()):
                        d=documents[h];body_pair_count.add((h,sid))
                        for title in d['titles']:
                            pair=pairs.setdefault((title,sid),{'r16_visible_source_title':title,
                                'other_corpus':corpus,'other_source_id':sid,'other_group_id':s['group_id'],
                                'other_task':s['task'],'other_answer_count':len(s['response_ids']),
                                'r16_occurrences':{},'r16_body_sha256':set(),'shared20_keys':set(),'examples':[]})
                            pair['r16_body_sha256'].add(h)
                            for occurrence in d['occurrences']:
                                if occurrence['visible_title']==title:
                                    pair['r16_occurrences'][occurrence['row_id']]=occurrence
                            if key not in pair['shared20_keys'] and len(pair['examples'])<3:
                                pair['examples'].append({'normalized20_words':normalized,'r16_body_sha256':h,
                                    'r16_start':original_a,'r16_end':original_b,'r16_quote':d['text'][original_a:original_b],
                                    'other_part_index':part_index,'other_start':other_a,'other_end':other_b,
                                    'other_quote':body[other_a:other_b]})
                            pair['shared20_keys'].add(key)
        serialized=[]
        for key,pair in sorted(pairs.items()):
            pair['r16_occurrences']=sorted(pair['r16_occurrences'].values(),key=lambda x:x['row_id'])
            pair['r16_body_sha256']=sorted(pair['r16_body_sha256'])
            pair['distinct_shared20_grams']=len(pair.pop('shared20_keys'))
            serialized.append(pair)
        matches.extend(serialized)
        corpora[corpus]={'answers':answer_count,'source_ids':len(sources),
            'source_groups':len({s['group_id'] for s in sources.values()}),'evidence_parts_scanned':evidence_parts,
            'unique_evidence_part_texts':len(unique_piece_hashes),'deduplicated_title_source_pairs':len(pairs),
            'deduplicated_body_source_pairs':len(body_pair_count),
            'matched_R16_rows':len({r['row_id'] for p in serialized for r in p['r16_occurrences']}),
            'matched_R16_questions':len({r['question_id'] for p in serialized for r in p['r16_occurrences']}),
            'matched_other_source_ids':len({p['other_source_id'] for p in serialized}),
            'matches':serialized}
    assert snap=={k:{'path':str(p.resolve()),'sha256':sha(p)} for k,p in INPUTS.items()}
    result={'status':'complete_read_only_check','scope':'Only R16 dataset_train and QA fit/auxiliary candidate train evidence. No original validation/test or sealed QA test access.',
        'normalization':'Unicode regex \\w+ words, casefold, punctuation/whitespace ignored; exactly20 consecutive normalized words, no stemming or semantic expansion.',
        'boundary_policy':'R16 visible passage bodies separately; QA passage1/2/3 bodies separately; Summary body and Data2txt original serialized evidence body separately. Never join separate QA passages.',
        'source_pair_dedup':'Unique visible R16 title + compared source_id within each corpus; repeated conditions/generators appear only as occurrence metadata. Body hash/source_id pairs also counted.',
        'R16':{'answers':len(rowids),'questions':len(questions),'groups':len(groups),'condition_counts':dict(conditions),
            'passage_occurrences':occurrences,'unique_visible_body_texts':len(documents),
            'unique_visible_titles':len({title for d in documents.values() for title in d['titles']}),
            'unique_normalized20_keys':len(target),'visible_bodies_shorter_than20_words':short},
        'comparisons':corpora,'total_deduplicated_source_pairs':len(matches),'inputs':snap,
        'script_sha256':sha(__file__),'seconds':time.perf_counter()-start,'scores_or_labels_used_for_matching':False,
        'samples_deleted_or_split_changed':False,'models_run':False,'GPU_used':False,
        'limits':['No exact20-word match does not establish complete material/entity/event independence.',
                  'Shorter, paraphrased, translated or semantically shared facts are outside this check.',
                  'Visible titles provide source-pair labels, not proof that two texts with a shared title are the same real-world source.',
                  'Any matches are disclosed, not used to delete cases or alter metrics. All are already-exposed development materials.']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'MATERIAL_REUSE_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=['# R16 与语义训练资料的有限重用检查','',
        f"只检查 R16 已开放训练的 {len(rowids)} 答、{len(questions)} 问、{len(groups)} 组。比较实际可见 passage 正文，未使用分数或标签决定匹配。",'',
        '| 比较训练材料 | 回答 | 来源 | 材料组 | 去重来源对匹配 | 命中R16回答 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,c in corpora.items():report.append(f"| {name} | {c['answers']} | {c['source_ids']} | {c['source_groups']} | {c['deduplicated_title_source_pairs']} | {c['matched_R16_rows']} |")
    report+=['','规则：连续 20 个 Unicode 词，忽略大小写、标点及空白差别。QA 不跨不同检索 passage 拼接；反复出现的生成答案和 complete/partial 资料以来源对去重。',
        '',('未发现满足此规则的材料重用。' if not matches else '发现材料重用，精确来源对、原文范围和匹配片段见机器结果。没有因此删除样本或更改划分。'),
        '这不证明完整实体或事件独立：短于20词、改写、翻译及同事实不同表述都不在本检查范围。',
        '', '没有读取 R16 原 validation/test 或 QA 封存测试；没有训练、重新评分、重划分或改动原材料。']
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'matches':len(matches),'R16':result['R16'],
                     'comparisons':{k:{kk:vv for kk,vv in v.items() if kk!='matches'} for k,v in corpora.items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
