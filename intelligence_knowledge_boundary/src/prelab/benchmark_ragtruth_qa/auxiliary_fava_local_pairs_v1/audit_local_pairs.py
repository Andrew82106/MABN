"""Independent source-coordinate reconstruction for the exported train-only pairs."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,importlib.util,json,re,time

OUT=Path(__file__).resolve().parent;QA=OUT.parent;UP=QA/'auxiliary_fava_v2'
spec=importlib.util.spec_from_file_location('original_fava_parser',QA/'additional_data/fava_training/inspect_training.py')
parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
TAG=re.compile(r'<(/?)([A-Za-z_]+)>')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def load(p):return json.loads(Path(p).read_text('utf-8'))
def rows(p):return [json.loads(s) for s in Path(p).read_text('utf-8').splitlines()]
def save(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def scan(completion):
    # A separate streaming counter, not the exporter's recursive tree collector.
    stack=[];source_nodes=[];cursor=0;raw_end=0;next_id=0
    for m in TAG.finditer(completion):
        if not any(x['tag']=='mark' for x in stack):cursor+=len(completion[raw_end:m.start()])
        closing,tag=m.groups();assert tag in parser.TAGS
        if not closing:
            item={'tag':tag,'char_start':cursor,'raw_start':m.start(),'nested_type':any(x['tag'] in parser.TYPES for x in stack)}
            if tag in parser.TYPES:item['ordinal']=next_id;next_id+=1
            stack.append(item)
        else:
            item=stack.pop();assert item['tag']==tag
            if tag in parser.TYPES:
                item.update(char_end=cursor,raw_end=m.end());source_nodes.append(item)
        raw_end=m.end()
    assert not stack
    return sorted(source_nodes,key=lambda n:n['ordinal'])

def main():
    start=time.perf_counter();manifest=load(OUT/'manifest.json');protocol=load(OUT/'protocol.json')
    source_hashes={Path(name).as_posix():h for name,h in protocol['sources_sha256'].items()}
    assert load(OUT/'export_complete.json')['manifest_sha256']==sha(OUT/'manifest.json')
    for name,h in manifest['artifacts_sha256'].items():assert sha(OUT/name)==h,name
    for name,h in protocol['sources_sha256'].items():assert sha(QA/name)==h,name
    original_rows=rows(UP/'candidate_fit.jsonl');original={r['response_id']:r for r in original_rows}
    assert len(original)==7482
    raw=load(QA/'additional_data/fava_training/training.json')
    groups={r['group_id']:r for r in rows(UP/'candidate_material_group_index.jsonl')}
    expected={}
    for r in original_rows:
        completion=raw[r['raw_index']]['completion'];corrupt=parser.parse_markup(completion)[0]
        assert corrupt==r['original_response'] and digest(completion)==r['completion_sha256']
        group=groups[r['group_id']]
        assert not group['quarantined'] and r['raw_index'] in group['raw_indices'] and r['response_id'] in group['response_ids']
        for n in scan(completion):
            if n['nested_type'] or n['tag'] not in {'entity','relation'}:continue
            snippet=completion[n['raw_start']:n['raw_end']]
            opening=[m.group(2) for m in TAG.finditer(snippet) if not m.group(1)]
            if opening.count('mark')!=1 or opening.count('delete')!=1 or sum(t in parser.TYPES for t in opening)!=1:continue
            bad,preferred,*_=parser.parse_markup(snippet)
            if not bad or not preferred or bad==preferred or not (1<=len(bad.split())<=3 and 1<=len(preferred.split())<=3):continue
            assert corrupt[n['char_start']:n['char_end']]==bad
            pid=f"{r['response_id']}__typed_{n['ordinal']:03d}"
            expected[pid]=(n,bad,preferred,digest(snippet))
    pairs=rows(OUT/'candidate_pairs.jsonl');input_rows=rows(OUT/'model_inputs.jsonl');prov_rows=rows(OUT/'provenance.jsonl')
    inputs={r['input_id']:r for r in input_rows};prov={r['pair_id']:r for r in prov_rows}
    assert len(inputs)==len(input_rows)==20080 and len(pairs)==len(prov_rows)==len(expected)==10040
    assert set(expected)=={p['pair_id'] for p in pairs}==set(prov)
    material_answers=defaultdict(set);answer_pairs=Counter();types=Counter();flags=[];used_inputs=set();checks=Counter()
    for p in pairs:
        pid=p['pair_id'];r=original[p['source_response_id']];n,bad,preferred,nodehash=expected[pid];pv=prov[pid]
        for k in ('raw_index','source_id','group_id'):assert p[k]==r[k]==pv[k]
        assert p['type']==pv['author_type']==n['tag']
        assert p['outside_targets']=='unknown_no_new_supervision' and p['human_gold'] is False and p['synthetic'] is True
        assert p['preference_is_factual_certificate'] is False
        assert pv['typed_node_preorder_index']==n['ordinal'] and pv['completion_node_char_span']==[n['raw_start'],n['raw_end']]
        assert pv['completion_node_sha256']==nodehash and pv['completion_sha256']==r['completion_sha256']
        assert pv['upstream_candidate_sha256']==source_hashes['auxiliary_fava_v2/candidate_fit.jsonl']
        assert pv['upstream_manifest_sha256']==manifest['upstream_manifest_sha256']
        assert pv['source_isolation']=='inherited_from_frozen_auxiliary_fava_v2_not_rescanned'
        assert pv['source_isolation_report_sha256']==source_hashes['auxiliary_fava_v2/SOURCE_ISOLATION_REPORT.json']
        assert pv['whole_answer_correctness_label_created'] is False
        targets=[p['original']['target'],p['preferred']['target']]
        assert [(t['start'],t['end'],t['text']) for t in targets]==[(n['char_start'],n['char_end'],bad),(n['char_start'],n['char_start']+len(preferred),preferred)]
        original_text=r['original_response'];s,e=n['char_start'],n['char_end']
        expected_texts=(original_text,original_text[:s]+preferred+original_text[e:])
        for side,text in zip(('original','preferred'),expected_texts):
            version=p[side];ir=inputs[version['input_id']];used_inputs.add(version['input_id'])
            assert set(ir)=={'input_id','model_inputs'} and set(ir['model_inputs'])=={'retrieved_passages','question','response'}
            mi=ir['model_inputs'];assert mi=={'retrieved_passages':r['retrieved_passages'],'question':'','response':text}
            assert digest(text)==version['response_sha256'] and digest(canonical(mi))==version['model_inputs_sha256']
            assert text[version['target']['start']:version['target']['end']]==version['target']['text']
            assert not any(m.group(2) in parser.TAGS for value in mi.values() for m in TAG.finditer(value))
        assert expected_texts[1][:s]==original_text[:s] and expected_texts[1][s+len(preferred):]==original_text[e:]
        assert expected_texts[1][:s]+bad+expected_texts[1][s+len(preferred):]==original_text
        refs,source_answer=parser.prompt_parts(raw[r['raw_index']]['prompt']);assert source_answer==original_text
        assert len(refs)==5 and [digest(x) for x in refs]==r['reference_text_sha256']==pv['reference_text_sha256']
        empties=[j+1 for j,x in enumerate(refs) if not x.strip()];reasons=[]
        if len(empties)==5:reasons.append('all_reference_bodies_empty_or_whitespace')
        if not any(c.isalnum() for x in refs for c in x):reasons.append('all_reference_bodies_without_alnum')
        if not any(c.isalnum() for c in bad):reasons.append('original_target_without_alnum')
        if not any(c.isalnum() for c in preferred):reasons.append('preferred_target_without_alnum')
        if not original_text.strip():reasons.append('original_response_empty_or_whitespace')
        el=p['eligibility'];assert el['reasons']==reasons and el['structural_eligible']==(not reasons) and el['empty_reference_ordinals']==empties
        if reasons:flags.append({'pair_id':pid,'raw_index':r['raw_index'],'original_target':bad,'preferred_target':preferred,'reasons':reasons,'retained':True})
        material_answers[p['group_id']].add(p['source_response_id']);answer_pairs[p['source_response_id']]+=1;types[p['type']]+=1
        checks['all_source_slice_patch_reverse_input_whitelist_passed']+=1
    assert len(used_inputs)==20080 and len(material_answers)==5238 and len(answer_pairs)==5275
    masses=Counter()
    for p in pairs:
        sw=p['suggested_base_weight'];na=len(material_answers[p['group_id']]);npair=answer_pairs[p['source_response_id']]
        assert sw['denominator']==5238*na*npair and sw['numerator']==1 and sw['value']==1/sw['denominator']
        assert sw['original_answers_in_group']==na and sw['pairs_in_original_answer']==npair
        masses[p['group_id']]+=sw['value']
    assert max(abs(v-1/5238) for v in masses.values())<1e-15 and abs(sum(masses.values())-1)<1e-10
    ag=rows(OUT/'original_answer_index.jsonl');gg=rows(OUT/'material_group_index.jsonl')
    assert {x['source_response_id']:x['pairs'] for x in ag}==answer_pairs
    assert {x['group_id']:set(x['source_response_ids']) for x in gg}==dict(material_answers)
    assert types=={'entity':5335,'relation':4705} and len(flags)==4
    save('STRUCTURAL_FLAGS.json',{'count':len(flags),'all_retained':True,'flags':flags})
    report={'passed':True,'verification':'Separate streaming markup source counter plus original parser, not exporter tree-collector replay.',
        'pairs':10040,'input_rows':20080,'original_answers':5275,'material_groups':5238,'types':dict(types),
        'checks':dict(checks),'structurally_eligible':10036,'flagged_but_retained':4,'all_candidate_nodes_covered_once':True,
        'original_group_membership_exact':True,'suggested_group_mass_equal':True,'outside_targets_unknown':True,
        'model_input_whitelist_exact':True,'whole_answer_correctness_labels_created':False,
        'source_isolation_inherited_not_rescanned':True,'sources_and_export_hashes_passed':True,
        'manifest_sha256':sha(OUT/'manifest.json'),'audit_code_sha256':sha(__file__),
        'GPU_used':False,'tokenizer_loaded':False,'trained':False,'QA_gold_cal_test_read':False,'seconds':time.perf_counter()-start}
    save('INDEPENDENT_QUALITY_CHECK.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
