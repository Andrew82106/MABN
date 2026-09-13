"""Train-only release inspection and provenance-preserving normalization; no training."""
from pathlib import Path
from collections import Counter,defaultdict
import difflib,hashlib,json,re,time

OUT=Path(__file__).resolve().parent
CORPORA=('DialogSum','SAMSum')
WORD=re.compile(r"\w+(?:['’]\w+)*|[^\w\s]",re.UNICODE)

def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def fsha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,r):(OUT/name).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n','utf-8')
def savel(name,rows):(OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),'utf-8')
def dist(xs):
    xs=sorted(xs)
    return {'n':len(xs),'sum':sum(xs),'min':xs[0] if xs else None,'median':xs[len(xs)//2] if xs else None,'p95':xs[min(len(xs)-1,int(.95*len(xs)))] if xs else None,'max':xs[-1] if xs else None}

def lev(a,b):
    """Unit-cost exact token Levenshtein; count optimal edit paths capped at2.

    This is a diagnostic alignment, NOT author span annotation or FERRANTI.
    """
    n,m=len(a),len(b);d=[list(range(m+1))];ways=[[1]*(m+1)]
    for i in range(1,n+1):
        row=[i]+[0]*m;w=[1]+[0]*m
        for j in range(1,m+1):
            costs=(d[-1][j-1]+int(a[i-1]!=b[j-1]),d[-1][j]+1,row[j-1]+1)
            v=min(costs);row[j]=v
            w[j]=min(2,(ways[-1][j-1] if costs[0]==v else 0)+(ways[-1][j] if costs[1]==v else 0)+(w[j-1] if costs[2]==v else 0))
        d.append(row);ways.append(w)
    i,j=n,m;edits=[]
    if ways[n][m]==1:
        ops=[]
        while i or j:
            opts=[]
            if i and j and d[i-1][j-1]+int(a[i-1]!=b[j-1])==d[i][j]:opts.append(('equal' if a[i-1]==b[j-1] else 'replace',i-1,i,j-1,j))
            if i and d[i-1][j]+1==d[i][j]:opts.append(('delete',i-1,i,j,j))
            if j and d[i][j-1]+1==d[i][j]:opts.append(('insert',i,i,j-1,j))
            assert len(opts)==1
            op=opts[0];ops.append(op);i,j=op[1],op[3]
        for op in reversed(ops):
            if op[0]=='equal':continue
            if edits and edits[-1]['a_end']==op[1] and edits[-1]['b_end']==op[3]:
                edits[-1]['a_end']=op[2];edits[-1]['b_end']=op[4];edits[-1]['operations'].append(op[0])
            else:edits.append({'a_start':op[1],'a_end':op[2],'b_start':op[3],'b_end':op[4],'operations':[op[0]]})
    return d[n][m],ways[n][m],edits

def charbound(matches,start,end,text):
    if start==end:
        q=matches[start].start() if start<len(matches) else len(text)
        return q,q
    # Include the gap following an edited block so insertion/deletion can be
    # replayed without dropping the separator before the next unchanged token.
    # The exact full-string patch below still rejects any unrelated rewriting.
    return matches[start].start(),matches[end].start() if end<len(matches) else len(text)

def run():
    tick=time.time();c=Counter();types=Counter();type_answers=Counter();bycorpus={};pairs=[];failures=[];groups=[];examples=[];edit_lengths=[];human_lengths=[];costs=[];single_predicate=[]
    docs=defaultdict(list);pure_pairs=[]
    for corpus in CORPORA:
        path=OUT/f'official_data__{corpus}__train.json';data=json.loads(path.read_text('utf-8'));assert len(data)==300
        seen=set();bc=Counter()
        for record in data:
            assert set(record)=={'id','dialogue','references','model_summaries'} and record['id'] not in seen
            seen.add(record['id']);material=sha(record['dialogue']);gid='dialsumm_material_'+material
            docs[material].append({'corpus':corpus,'id':record['id']})
            groups.append({'corpus':corpus,'original_dialogue_id':record['id'],'official_fec_split':'train','material_sha256':material,'group_id':gid,'model_names':list(record['model_summaries'])})
            assert set(record['model_summaries'])=={'BART','UniLM','MV-BART','CODS'}
            for model,r in record['model_summaries'].items():
                assert set(r)=={'original_summary','modified','consistency','error_categories'}
                rid=f'dialsumm_{corpus}_{record["id"]}_{model}';a,b=r['original_summary'],r['modified'];changed=a!=b
                c['answers']+=1;bc['answers']+=1;c['human_inconsistent']+=not r['consistency'];bc['human_inconsistent']+=not r['consistency'];c['changed']+=changed;bc['changed']+=changed
                c['inconsistent_but_unchanged']+=int(not r['consistency'] and not changed)
                c['consistent_but_changed']+=int(r['consistency'] and changed)
                c['inconsistent_without_spans']+=int(not r['consistency'] and not r['error_categories'])
                c['consistent_with_spans']+=int(r['consistency'] and bool(r['error_categories']))
                labs=[]
                for original in r['error_categories']:
                    assert set(original)=={'start','text','type'}
                    s=original['start'];e=s+len(original['text']);exact=0<=s<e<=len(a) and a[s:e]==original['text']
                    c['human_spans']+=1;c['human_spans_exact']+=exact;types[original['type']]+=1
                    human_lengths.append(len(original['text']))
                    if not exact:failures.append({'response_id':rid,'original_label':original,'reason':'character_slice_mismatch'})
                    labs.append({**original,'end':e,'end_derivation':'original_start + len(original_span_text)','character_slice_exact':exact})
                for t in {x['type'] for x in labs}:type_answers[t]+=1
                ma=list(WORD.finditer(a));mb=list(WORD.finditer(b));ta=[x.group() for x in ma];tb=[x.group() for x in mb]
                if changed:
                    cost,ways,edits=lev(ta,tb);costs.append(cost)
                    c['changed_unique_token_edit_path']+=ways==1;c['changed_ambiguous_token_edit_path']+=ways>1;c['changed_formatting_only']+=cost==0
                else:cost,ways,edits=0,1,[]
                for edit in edits:
                    s,e=charbound(ma,edit['a_start'],edit['a_end'],a);u,v=charbound(mb,edit['b_start'],edit['b_end'],b)
                    edit.update({'original_char_start':s,'original_char_end':e,'modified_char_start':u,'modified_char_end':v,
                                 'original_text':a[s:e],'modified_text':b[u:v],'original_token_count':edit['a_end']-edit['a_start'],'modified_token_count':edit['b_end']-edit['b_start']})
                    edit['overlapping_original_human_types']=sorted({l['type'] for l in labs if s<l['end'] and e>l['start']})
                    edit_lengths.append(max(e-s,v-u))
                # Strong string-only single-change gate: swapping one derived span reproduces modified verbatim.
                pure=False;localpred=False;smallpred=False;paired_scope=None
                if changed and cost>0 and ways==1 and len(edits)==1:
                    ed=edits[0];s,e=ed['original_char_start'],ed['original_char_end'];u,v=ed['modified_char_start'],ed['modified_char_end']
                    pure=a[:s]+b[u:v]+a[e:]==b
                    c['one_unique_edit_block']+=1;c['one_block_exact_fulltext_patch']+=pure
                    short=pure and 1<=ed['original_token_count']<=3 and 1<=ed['modified_token_count']<=3
                    c['one_block_exact_short_replacement']+=short
                    localpred=short and len(labs)==1 and labs[0]['type']=='PredE' and 'PredE' in ed['overlapping_original_human_types']
                    c['one_block_exact_short_single_PredE_replacement']+=localpred
                    small=pure and 0<=ed['original_token_count']<=3 and 0<=ed['modified_token_count']<=3
                    smallpred=small and len(labs)==1 and labs[0]['type']=='PredE' and 'PredE' in ed['overlapping_original_human_types']
                    c['one_block_exact_small_single_PredE_including_insertion_deletion']+=smallpred
                    if pure:pure_pairs.append(rid)
                    if smallpred:single_predicate.append(rid)
                if smallpred:
                    lo=0
                    while lo<min(len(a),len(b)) and a[lo]==b[lo]:lo+=1
                    suffix=0
                    while suffix<min(len(a)-lo,len(b)-lo) and a[-suffix-1]==b[-suffix-1]:suffix+=1
                    hi=len(a)-suffix;lab=labs[0]
                    if lab['start']<=lo and hi<=lab['end']:
                        end_b=lab['end']+len(b)-len(a)
                        assert a[:lab['start']]==b[:lab['start']] and a[lab['end']:]==b[end_b:]
                        paired_scope={'original_start':lab['start'],'original_end':lab['end'],
                                      'modified_start':lab['start'],'modified_end':end_b,
                                      'original_text':a[lab['start']:lab['end']],'modified_text':b[lab['start']:end_b],
                                      'scope_alignment_only_not_a_new_truth_label':True}
                        c['small_single_PredE_original_human_scope_maps_exactly']+=1
                normalized={'response_id':rid,'corpus':corpus,'official_fec_split':'train','original_dialogue_id':record['id'],'source_group':gid,
                     'material_sha256':material,'retrieved_passages':record['dialogue'],'question':'','original_response':a,
                     'human_corrected_response':b,'author_original_consistency':r['consistency'],'original_human_error_spans':labs,
                     'original_response_sha256':sha(a),'human_corrected_response_sha256':sha(b),'generator':model,
                     'generator_metadata_not_feature':True,'original_reference_summaries_not_input':record['references'],
                     'change_analysis':{'changed':changed,'tokenizer':'regex words with apostrophe plus punctuation; no lowercasing or whitespace rewrite',
                       'unit_token_edit_distance':cost,'optimal_path_count_capped2':ways,'unique_path_derived_edits':edits,
                       'exact_single_span_patch':pure,'short_single_PredE_replacement_candidate':localpred,'small_single_PredE_edit_candidate':smallpred,
                       'paired_original_human_scope':paired_scope,
                       'derived_alignment_not_new_human_annotation':True},
                     'corrected_whole_answer_guarantee':'Author human correction; not an independent proof of every fact. This review does not relabel the corrected answer.',
                     'model_input_fields':['retrieved_passages','question','original_response'],'source_isolation_with_existing_QA':'not yet performed',
                     'training_status':'review_only_not_merged_or_trained','data_license':'not specified in author archive or repository root'}
                pairs.append(normalized)
                if (corpus,record['id'],model) in {('DialogSum','test_13','MV-BART'),('SAMSum','13729857','UniLM'),('SAMSum','13717092','UniLM'),('DialogSum','test_256','UniLM')}:
                    examples.append(normalized)
        bycorpus[corpus]={'dialogues':len(data),**dict(bc)}
    assert c['answers']==2400 and len(groups)==600 and c['human_spans']==1379 and not failures
    savel('train_pairs_review.jsonl',pairs);savel('train_dialogue_groups.jsonl',groups);savel('alignment_failures.jsonl',failures);save('TRAIN_EXAMPLES.json',examples)
    save('STRUCTURAL_SINGLE_PREDICATE_IDS.json',{'rows':single_predicate,'selection':'Exact single derived edit with0–3tokens on each side, including insertion/deletion, overlapping the sole original human PredE span; not a new truth judgment.'})
    report={'status':'complete_train_only_review','counts':dict(c),'by_corpus':bycorpus,'original_error_span_type_counts':dict(types),'answers_by_original_error_type':dict(type_answers),
      'original_dialogue_groups':len(groups),'exact_material_groups':len(docs),'duplicate_material_groups':[v for v in docs.values() if len(v)>1],
      'unique_path_edit_char_length_max_side':dist(edit_lengths),'original_human_error_span_character_lengths':dist(human_lengths),'changed_answer_unit_token_edit_distances':dist(costs),
      'small_single_PredE_edit_material_groups':len({r['source_group'] for r in pairs if r['response_id'] in set(single_predicate)}),
      'small_single_PredE_exact_scope_material_groups':len({r['source_group'] for r in pairs if r['change_analysis']['paired_original_human_scope'] is not None}),
      'split_verification':'Only author train files opened. Paper declares300/100/100 dialogue split per corpus; val/test IDs or content not read, so cross-split disjointness not independently rechecked.',
      'original_dataset_id_note':'DialogSum IDs can start with test_ because FEC dataset was constructed from original summarization test material and defines its own new train split. This is FEC train, not FEC/QA heldout evaluation.',
      'license_status':'No dataset-specific license in archive README, archive member list, root README or repository API. errant/LICENSE.md is inherited MIT software license, not assumed to license dialogue annotations.',
      'new_model_or_training_labels':False,'GPU_used':False,'local_QA_cal_test_read':False,'FEC_val_test_total_content_read':False,
      'seconds':time.time()-tick,'input_sha256':{f'official_data__{corpus}__train.json':fsha(OUT/f'official_data__{corpus}__train.json') for corpus in CORPORA}}
    save('TRAIN_REVIEW.json',report)
    names=['train_pairs_review.jsonl','train_dialogue_groups.jsonl','alignment_failures.jsonl','TRAIN_EXAMPLES.json','STRUCTURAL_SINGLE_PREDICATE_IDS.json','TRAIN_REVIEW.json']
    save('manifest.json',{'status':'train_only_provenance_normalization_not_ready_for_training','answers':2400,'groups':len(docs),'license':report['license_status'],
                         'source_isolation_with_QA_completed':False,'artifacts_sha256':{n:fsha(OUT/n) for n in names},'protocol_sha256':fsha(OUT/'PROTOCOL.json')})
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':run()
