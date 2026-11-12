"""Reconcile explicit independent judgments and export labels, without model fitting."""
from collections import Counter, defaultdict
import argparse
import importlib.util
import json
from pathlib import Path

import annotation16 as ann
ROOT = ann.ROOT
readl, sha = ann.readl, ann.sha
spec = importlib.util.spec_from_file_location('r16_original_alignment', ROOT.parent/'round8_token_localization/src/evaluate8.py')
metric = importlib.util.module_from_spec(spec); spec.loader.exec_module(metric)
save, savel = metric.save, metric.savel


def judgments(pattern, records):
    result, files = {}, {}
    for p in sorted((ROOT/'data/annotations').glob(pattern)):
        doc = json.loads(p.read_text('utf-8')); files[p.relative_to(ROOT).as_posix()] = sha(p)
        for d in doc['decisions']:
            iid = d['item_id']; assert iid not in result, ('Duplicate judgment', iid)
            r = ann.record(d, doc['annotator'], records)
            r['reviewed_safe_refusal'] = bool(d.get('reviewed_safe_refusal', False))
            if r['reviewed_safe_refusal']:
                assert r['original_stance'] == 'abstained' and not r['risk_spans']
            result[iid] = r
    return result, files


def decision_key(r):
    return (r['original_stance'], r['original_risk'], r['evidence_relation'], r['localization_status'],
            r['reviewed_safe_refusal'], [(s['start'],s['end']) for s in r['risk_spans']])


def reconcile(export=False):
    inputs, records = ann.load_rows()
    initial, f1 = judgments('initial_*.json', records)
    review, f2 = judgments('independent_*.json', records)
    adjudication, f3 = judgments('adjudication_*.json', records)
    common = set(initial) & set(review)
    actor=lambda label: label['annotator'].split(':')[0]
    assert all(actor(initial[i]) != actor(review[i]) for i in common)
    for iid,a in adjudication.items():
        if iid in common:
            assert actor(a) not in (actor(initial[iid]),actor(review[iid])), ('Adjudication needs a third reviewer',iid)
    differences = sorted(i for i in common if decision_key(initial[i]) != decision_key(review[i]))
    save(ROOT/'data/reviews/disagreements.json', {
        'generated':len(records), 'initial':len(initial), 'independent':len(review),
        'compared':len(common), 'exact_decision_disagreements':len(differences),
        'unadjudicated': [i for i in differences if i not in adjudication],
        'decisions':[{'item_id':i, 'initial':initial[i], 'independent':review[i]} for i in differences]})
    if not export:
        print('JUDGMENTS', len(initial), len(review), 'DISAGREEMENTS', len(differences),
              'UNADJUDICATED', len(set(differences)-set(adjudication))); return
    assert not (ROOT/'data/annotation_freeze.json').exists(), 'Completed labels are immutable'
    assert len(records) == len(inputs) and set(initial) == set(review) == set(records)
    assert set(differences) <= set(adjudication)
    assert set(adjudication) <= set(records)
    manifest = json.loads((ROOT/'data/generation_manifest.json').read_text('utf-8'))
    assert manifest['complete'] and manifest['generated'] == len(inputs)
    frozen = json.loads((ROOT/'data/input_freeze.json').read_text('utf-8'))
    for name, checksum in frozen['files_sha256'].items(): assert sha(ROOT/name) == checksum, name
    old = json.loads((ROOT/'data/legacy_files_snapshot.json').read_text('utf-8'))
    for name, checksum in old['files_sha256'].items(): assert sha(ROOT.parent/name) == checksum, name
    output = []
    for row in inputs:
        iid = row['row_id']+'__1'; a = dict(adjudication.get(iid, initial[iid]))
        a.update(category=row['category'], independent_reviewer=review[iid]['annotator'],
                 initial_annotator=initial[iid]['annotator'], adjudicated=iid in adjudication,
                 initial_review_agree=iid not in differences,
                 answer_gold=0 if a['reviewed_safe_refusal'] else a['original_risk'])
        assert sha(ROOT/'data/generation_records'/(row['row_id']+'.json')) == manifest['record_sha256'][row['row_id']]
        output.append(a)
    files = dict(f1, **f2, **f3)
    safe_files, stats = {}, {}
    for split in ('train', 'validation', 'test'):
        selected = [a for a in output if a['split'] == split]
        ap = ROOT/'data'/f'annotations_{split}.jsonl'; savel(ap, selected)
        ready=[]
        for a in selected:
            row,g,_,checksum=records[a['item_id']]
            ready.append({'input':row,'response':g['response'],
                          'response_token_ids':g['response_token_ids'],
                          'response_token_offsets':g['response_token_offsets'],
                          'annotation':a,'source_generation_sha256':checksum})
        ready_path=ROOT/'data'/f'dataset_{split}.jsonl';savel(ready_path,ready)
        files[ready_path.relative_to(ROOT).as_posix()]=sha(ready_path)
        safe = [a['item_id'] for a in selected if a['reviewed_safe_refusal']]
        sp = ROOT/'data'/f'safe_refusals_{split}.json'
        save(sp, {'status':'reviewed_frozen','split':split,'source_annotation_sha256':sha(ap),'safe_refusal_item_ids':safe})
        safe_files[split] = sha(sp)
        tokens, windows, regions = [], [], []
        for a in selected:
            row, g, item, _ = records[a['item_id']]
            item = dict(item, **{k:row[k] for k in ('row_id','question_id','group_id','split','condition','category')})
            gold = {a['item_id']:a}
            original = {a['item_id']:{'annotation':{'stance':a['original_stance'],'risk':a['original_risk']}}}
            tt, rr = metric.align_row(g, [item], gold, original)
            for t in tt: t['category'] = row['category']
            tokens.extend(tt); regions.extend(rr)
            if a['localization_status'] != 'resolved': continue
            indices = [j for j,(left,right) in enumerate(g['response_token_offsets']) if right > a['start'] and left < a['end']]
            assert indices and all(j == i+1 for i,j in zip(indices,indices[1:]))
            starts = range(len(indices)-4+1) if len(indices) >= 4 else [0]
            for start in starts:
                raw = indices[start:start+4]; group = [tt[j] for j in raw]
                assert not any(t['lexical'] and not t['main_eligible'] for t in group)
                eligible = [t for t in group if t['main_eligible']]
                if not eligible: continue
                left = max(a['start'], group[0]['start']); right = min(a['end'],group[-1]['end'])
                windows.append({'window_key':a['item_id']+f'__w4__start{raw[0]}',
                                **{k:row[k] for k in ('row_id','question_id','group_id','split','condition','category')},
                                'item_ids':[a['item_id']], 'width':4, 'actual_width':len(raw),'short_window':len(raw)<4,
                                'raw_token_indices':raw,'token_keys':[t['token_key'] for t in eligible],
                                'start':left,'end':right,'text':g['response'][left:right],
                                'gold':int(any(t['gold'] for t in eligible))})
        for name, values in [('tokens',tokens),('windows_k4',windows),('risk_regions',regions)]:
            p=ROOT/'data'/f'{name}_{split}.jsonl'; savel(p,values); files[p.relative_to(ROOT).as_posix()]=sha(p)
        files[ap.relative_to(ROOT).as_posix()] = sha(ap); files[sp.relative_to(ROOT).as_posix()] = sha(sp)
        stats[split] = {'questions':len({a['question_id'] for a in selected}),
                        'event_or_subject_groups':len({a['group_id'] for a in selected}), 'answers':len(selected),
                        'supported_answers':sum(a['original_risk']==0 for a in selected),
                        'risky_answers':sum(a['original_risk']==1 for a in selected),
                        'safe_refusals':len(safe), 'unresolved_or_other':sum(a['answer_gold'] is None for a in selected),
                        'localization_answers':sum(a['localization_status']=='resolved' for a in selected),
                        'eligible_tokens':sum(t['main_eligible'] for t in tokens),
                        'risk_tokens':sum(t['gold']==1 for t in tokens),
                        'windows_4':len(windows),'risk_windows_4':sum(w['gold'] for w in windows),
                        'question_categories':dict(Counter(a['category'] for a in selected if a['condition']=='complete'))}
    pp=ROOT/'data/question_label_policy.json';save(pp,{'status':'reviewed_frozen','reviewed_safe_refusal_files_sha256':safe_files,
        'risk_definition':'unsupported or contradicted claim relative to current displayed evidence',
        'safe_refusal_rule':'explicit independent review; answer 0, excluded from localization',
        'nonrisk_token_meaning':'outside annotated risk spans, not individually verified factual truth'})
    files[pp.relative_to(ROOT).as_posix()] = sha(pp)
    refs=readl(ROOT/'data/references.jsonl');snapshots={r['candidate_id']:r for r in readl(ROOT/'data/source_snapshots.jsonl')}
    save(ROOT/'results/dataset_statistics.json',{'new':stats,'initial_independent_disagreements':len(differences),
        'source_topic_questions':dict(Counter(snapshots[r['candidate_id']]['official_category'] for r in refs)),
        'evidence_relations':dict(Counter(a['evidence_relation'] for a in output)),
        'condition_labels':{cond:dict(Counter(str(a['answer_gold']) for a in output if a['condition']==cond)) for cond in ('complete','partial')},
        'human_gold':False, 'probe_training_performed':False, 'counts_are_not_performance_metrics':True})
    old_inputs=readl(ROOT.parent/'round10_dual_granularity/data/inputs.jsonl')
    combined=[]
    for tag, rows, prefix in [('legacy_round10',old_inputs,'../round10_dual_granularity'),('new_round16',inputs,'.')]:
        for row in rows:
            combined.append({k:row[k] for k in ('row_id','question_id','group_id','split','condition','category')})
            combined[-1].update(source_cohort=tag,input_file=f'{prefix}/data/inputs.jsonl',
                generation_file=f"{prefix}/data/generation_records/{row['row_id']}.json",
                annotation_file=f"{prefix}/data/annotations_{row['split']}.jsonl",
                training_eligible=row['split']=='train', fresh_holdout=tag=='new_round16' and row['split']!='train')
    savel(ROOT/'data/combined_index.jsonl',combined)
    save(ROOT/'data/combined_manifest.json',{'total_questions':len({r['question_id'] for r in combined}),
        'total_answers':len(combined),'split_questions':dict(Counter(r['split'] for r in combined if r['condition']=='complete')),
        'legacy_files_sha256':old['files_sha256'], 'combined_index_sha256':sha(ROOT/'data/combined_index.jsonl'),
        'fresh_evaluation':'Use new_round16 validation/test separately; legacy holdouts have already been exposed.',
        'grouping_limit':'New related sources/events are grouped; old group ids are preserved, not re-certified as globally independent.',
        'training_features':'New white-box features are not yet extracted. Retain exact generation IDs for later extraction.'})
    save(ROOT/'data/annotation_freeze.json',{'status':'assistant_annotated_independently_reviewed_frozen',
        'answers':len(output),'generation_manifest_sha256':sha(ROOT/'data/generation_manifest.json'),
        'input_freeze_sha256':sha(ROOT/'data/input_freeze.json'),'files_sha256':files,
        'human_gold':False,'predictions_viewed':False,'probe_training_performed':False})
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--export',action='store_true');a=p.parse_args();reconcile(a.export)
