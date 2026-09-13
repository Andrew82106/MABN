"""Separate coordinate integrity from optional metadata; no repaired examples."""
from pathlib import Path
from collections import Counter
import importlib.util
import json
import re
import hashlib
import pyarrow.parquet as pq

OUT=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def rows(n):return [json.loads(x) for x in (OUT/n).read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run():
    assert not (OUT/'diagnostic_details.json').exists()
    complete=read('complete.json')
    for n,h in complete['files_sha256'].items():assert sha(OUT/n)==h
    checks=rows('english_rows.jsonl')
    data={r['id']:r for r in pq.read_table(OUT/'train-00000-of-00001.parquet').to_pylist() if r['lang']=='en'}
    spec=importlib.util.spec_from_file_location('psilo_inventory',OUT/'audit_train.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    # Independent regexp reconstruction for every well formed markup row.
    tested=0
    for c in checks:
        r=data[c['id']];markup=r['annotated_span']
        if any(not e.startswith('empty_or_nonstring_') for e in c['errors']):continue
        clean=re.sub(r'\[/?HAL\]','',markup);pairs=[]
        for m in re.finditer(r'\[HAL\](.*?)\[/HAL\]',markup,re.S):
            start=len(re.sub(r'\[/?HAL\]','',markup[:m.start()]))
            pairs.append([start,start+len(m.group(1))])
        assert pairs==c['markup_projection_spans']
        assert (clean==r['llm_answer'])==c['projection_exact'];tested+=1
    assert audit.parse_markup('a[HAL]b\nc[/HAL]d')==('ab\ncd',[[1,4]],[])
    assert 'orphan_close' in audit.parse_markup('a[/HAL]b[HAL]c[/HAL]')[2]
    assert 'nested_open' in audit.parse_markup('[HAL]a[HAL]b[/HAL][/HAL]')[2]
    coordinate=[]; texts=[]
    for c in checks:
        tag_errors=[e for e in c['errors'] if not e.startswith('empty_or_nonstring_')]
        passed=not tag_errors and all(c[k] for k in ['labels_in_bounds','labels_ordered_nonoverlap','projection_exact','labels_equal_markup_positions'])
        coordinate.append({'id':c['id'],'strict_coordinate_pass':passed,'metadata_issues':[e for e in c['errors'] if e.startswith('empty_or_nonstring_')]})
        if passed:texts.extend(data[c['id']]['llm_answer'][a:b] for a,b in c['original_labels'])
    (OUT/'coordinate_status.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in coordinate),encoding='utf-8')
    dup=rows('same_material_question_answer_duplicates.jsonl')
    examples=[]
    for d in dup:
        examples.append({'kind':'identical_detector_input_different_released_labels','input_sha256':d['input_key_sha256'],
            'original_records':[data[rid] for rid in d['ids']]})
    for suffix in ['TinyLlama/TinyLlama-1.1B-Chat-v1.0_2','TinyLlama/TinyLlama-1.1B-Chat-v1.0_124','TinyLlama/TinyLlama-1.1B-Chat-v1.0_380','TinyLlama/TinyLlama-1.1B-Chat-v1.0_4']:
        r=data['psiloqa_'+suffix];examples.append({'kind':'readable_format_or_language_example','original_record':r,
            'check':next(c for c in checks if c['id']==r['id'])})
    save('review_examples.json',examples)
    numeric=Counter();same_text_wrong_location=0;empty_multiline=0
    for c in checks:
        r=data[c['id']]
        clean=re.sub(r'\[/?HAL\]','',r['annotated_span'])
        if c['empty_labels_but_HAL_positive']:
            empty_multiline+=any('\n' in clean[a:b] for a,b in c['markup_projection_spans'])
        if c['projection_exact'] and len(c['original_labels'])==len(c['markup_projection_spans']):
            for (a,b),(x,y) in zip(c['original_labels'],c['markup_projection_spans']):
                same_text_wrong_location+=(a,b)!=(x,y) and r['llm_answer'][a:b]==r['llm_answer'][x:y]
    good={c['id'] for c in coordinate if c['strict_coordinate_pass']}
    stats={
        'status':'completed_diagnostic_no_repair',
        'summary_interpretation':'summary.strict_exact_pass also requires all metadata strings nonempty. The coordinate-only integrity count below does not reject missing optional complexity. No rows were removed or rewritten.',
        'strict_coordinate_pass':len(good),'coordinate_failure_rows':len(checks)-len(good),
        'coordinate_pass_positive':sum(bool(data[r]['labels']) for r in good),
        'coordinate_pass_unmarked':sum(not data[r]['labels'] for r in good),
        'full_nonempty_schema_and_coordinate_pass':sum(c['strict_exact_pass'] for c in checks),
        'metadata_only_failure_rows':sum(c['id'] in good and bool(c['metadata_issues']) for c in coordinate),
        'markup_malformed_rows':sum(any(not e.startswith('empty_or_nonstring_') for e in c['errors']) for c in checks),
        'projection_not_exact_rows':sum(not c['projection_exact'] for c in checks),
        'label_positions_not_exact_rows':sum(not c['labels_equal_markup_positions'] for c in checks),
        'failure_categories_overlap':True,'same_text_wrong_occurrence_span_count':same_text_wrong_location,
        'empty_labels_HAL_positive_rows':sum(c['empty_labels_but_HAL_positive'] for c in checks),
        'empty_labels_HAL_positive_with_multiline_span_rows':empty_multiline,
        'coordinate_pass_span_statistics':audit.span_stats(texts),
        'all_three_duplicate_groups_have_different_labels':len(dup)==3 and all(d['different_label_lists'] for d in dup),
        'duplicate_input_answer_polarity_disagreement_groups':sum(len({bool(data[i]['labels']) for i in d['ids']})>1 for d in dup),
        'manual_example_limits':'Examples are format/language observations and apparent support conflicts, not new human labels or an estimated semantic annotation error rate.',
        'regex_independent_geometry_rows_checked':tested,
        'detector_input_if_future_authorized':['wiki_passage','question','llm_answer'],
        'golden_answer_is_annotation_provenance_only_never_detector_input':True,
        'no_models_no_GPU_no_training_no_newsplit':True,
    }
    save('diagnostic_details.json',stats)
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':run()
