"""Serialize root's explicit, pre-score adjudications; never infer gold from condition."""
from pathlib import Path
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import annotation10 as a


def read(path): return json.loads(Path(path).read_text('utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, data): Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n','utf-8')


def main():
    assert not (ROOT/'results/freeze10.json').exists()
    assert not (ROOT/'results/test_started10.json').exists()
    assert not (ROOT/'data/annotation_freeze.json').exists()
    folder=ROOT/'data/annotations'; merged={}; review_index={}; provenance={}
    for initial, reviewer in [('root','evaluation'), ('evaluation','redeep'), ('redeep','data'), ('data','root')]:
        p=folder/f'test_initial_{initial}.json'; q=folder/f'test_review_{reviewer}.json'
        doc=read(p); rev=read(q); assert rev['initial_file_sha256']==sha(p)
        decisions=doc['decisions']; reviews=rev.get('items',rev.get('reviews',[]))
        assert len(decisions)==len(reviews)==30
        rr={v['item_id']:v for v in reviews}; assert set(rr)=={d['item_id'] for d in decisions}
        for d in decisions:
            iid=d['item_id']; assert iid not in merged
            assert rr[iid]['source_generation_sha256']==d['source_generation_sha256']
            merged[iid]=copy.deepcopy(d); review_index[iid]=rr[iid]
        provenance[p.name]=sha(p); provenance[q.name]=sha(q)

    changes={
      'ragognize_train_0348__partial__1': {
          'evidence_relation':'unresolved','risk':None,'spans':[],'localization_status':'unresolved',
          'rationale':'Final blind adjudication: a 2024 film can mean release/film year without strictly determining theatrical premiere year. Independent reviewer identified this event-specific ambiguity; do not import the other condition or force a risk label.'},
      'ragognize_train_0081__partial__1': {
          'evidence_relation':'unresolved','risk':None,'spans':[],'localization_status':'unresolved',
          'rationale':'Final blind adjudication: offers support/resources may be ordinary paraphrase of a community group purpose, or distinct from a formal advocacy position. Initial and second judgments differ on this ambiguous relation, so retain unresolved; incompleteness alone is not risk.'},
      'ragognize_train_0121__partial__1': {
          'spans':['by caring for'],
          'rationale':'Final blind adjudication: source supports care for over 50 species and breeding. It does not establish care as the means of public education. Circle the unsupported causal/action link by caring for; keep the known animal-care quantities and breeding background uncolored.'},
      'ragognize_train_0597__partial__1': {
          'spans':['highlight when sign language is featured prominently'],
          'rationale':'Final blind adjudication: searchable catalog listings are given, but a specific prominence-highlighting criterion is not. Circle that added criterion and retain supported listings/film background.'},
      'ragognize_test_1482__partial__1': {
          'spans':['did not raise any amount', 'as it is already'],
          'rationale':'Final blind adjudication: no displayed target source establishes zero IPO financing. The answer additionally uses current public-listing status as a causal/temporal basis for no January 2024 financing. Circle the denial plus as it is already, retaining the source-supported publicly listed company description.'},
    }
    adjudications=[]
    for iid, patch in changes.items():
        old=copy.deepcopy(merged[iid]); merged[iid].update(patch)
        adjudications.append({'item_id':iid,'initial':old,'final':copy.deepcopy(merged[iid]),'detector_scores_viewed':False})
    save(folder/'test_adjudication_log.json', {'reviewer':'root','status':'adjudicated_before_fit_and_test',
        'initial_and_review_files_sha256':provenance,'all_120_reviewed':True,
        'independent_root_second_sha256':sha(folder/'test_second_root.json'),
        'changes':adjudications,'unchanged_unresolved_items':['ragognize_train_0472__partial__1','ragognize_train_0256__partial__1','ragognize_train_0583__partial__1'],
        'span_decisions_retained':'Keep no college as to include the unsupported causal link. Keep only Gender Recognition Bill as misassigned advocacy value; its displayed donor provisions remain uncolored.',
        'no_runtime_llm_judge_api':True,'human_gold':False})
    inputs,_=a.load_rows(); expected=[r['row_id']+'__1' for r in inputs if r['split']=='test']
    assert set(merged)==set(expected) and len(expected)==120
    a.add('data/annotations/test_adjudicated.json',[merged[i] for i in expected],annotator='root_round10_adjudicated_after_blind_review')
    a.serialize_test(['data/annotations/test_adjudicated.json'])

    safe_ids=[
       'ragognize_train_0085__partial__1','ragognize_train_0307__partial__1','ragognize_train_1060__partial__1',
       'ragognize_train_0107__partial__1','ragognize_train_0355__partial__1','ragognize_train_0609__partial__1',
       'ragognize_train_0693__partial__1','ragognize_train_1546__partial__1','ragognize_train_1329__partial__1',
       'ragognize_test_0064__partial__1','ragognize_test_1848__partial__1','ragognize_train_2000__partial__1',
       'ragognize_test_0850__partial__1','ragognize_train_0035__partial__1','ragognize_train_0182__partial__1']
    for iid in safe_ids:
        d=merged[iid]; rv=review_index[iid]
        assert d['stance']=='abstained' and d['risk'] is None
        assert rv.get('decision',rv.get('verdict'))=='approve'
    assert set(safe_ids)=={i for i,d in merged.items() if d['stance']=='abstained'}
    save(ROOT/'data/safe_refusals_test.json', {'schema':'round10-safe-refusal-review-v1','split':'test',
        'status':'reviewed_frozen','reviewer':'initial annotator plus independent reviewer; root final listed-response and rationale check',
        'safe_refusal_item_ids':safe_ids,'source_annotation_sha256':sha(ROOT/'data/annotations_test.jsonl'),
        'basis':'Every listed case explicitly reviewed from actual sources and full output twice. Root read all listed responses/rationales. They decline absent target details without unsupported factual additions; scope imprecision/task incompleteness is distinct from invented facts.',
        'token_labels_modified':False})
    save(ROOT/'data/question_label_policy.json', {'schema':'round10-question-label-policy-v1','status':'reviewed_frozen',
        'reviewed_safe_refusal_files_sha256':{s:sha(ROOT/f'data/safe_refusals_{s}.json') for s in ('train','validation','test')}})
    save(ROOT/'data/annotation_freeze.json', {'schema':'round10-annotation-freeze-v1','status':'frozen',
        'created_before_fit_and_test':True,'human_gold':False,'all_new_items_independently_reviewed':True,
        'canonical_spans_sha256':{s:sha(ROOT/f'data/annotations_{s}.jsonl') for s in ('train','validation','test')},
        'question_label_policy_sha256':sha(ROOT/'data/question_label_policy.json'),
        'annotation_provenance_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in folder.glob('test_*.json')},
        'serializer_and_explicit_adjudication_script_sha256':sha(Path(__file__))})
    from collections import Counter
    print('FINAL_TEST_LABELS',Counter((d['stance'],d['risk'],d.get('localization_status')) for d in merged.values()))


if __name__=='__main__': main()
