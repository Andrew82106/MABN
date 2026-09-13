"""Fixed-order qualitative packets from original FIT only; no inference/refit."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODEL = ROOT / 'results/citation_alignment_lr_v1'
KINDS = ('Evident Conflict', 'Subtle Conflict', 'Evident Baseless Info')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    selected = json.loads((MODEL/'summary.json').read_text())['selected']['harp_claim__two_scores_and_citation']
    score_path = MODEL/(selected['candidate']+'_scores.npz')
    assert sha(score_path) == selected['scores_sha256']
    paths = [ROOT/'data'/n for n in ('fit.jsonl', 'tokens_fit.jsonl', 'windows_k4_fit.jsonl')]
    rows = {}; line_numbers = {}
    for i, line in enumerate(paths[0].open(encoding='utf-8'), 1):
        row = json.loads(line)
        assert row['partition'] == 'fit' and row['official_split'] == 'train'
        rows[row['response_id']] = row; line_numbers[row['response_id']] = i
    tokens = [json.loads(l) for l in paths[1].open(encoding='utf-8')]
    windows = [json.loads(l) for l in paths[2].open(encoding='utf-8')]
    assert len(rows) == len(tokens) == 634 and len(windows) == 168123
    # Producer concatenates fit then calibration. Only fit scores are used.
    with np.load(score_path) as f:
        scores = f['window_scores'][:len(windows)].copy()
    threshold = selected['thresholds']['window']['threshold']
    pred = scores >= threshold
    labels = np.array([w['label'] for w in windows])
    counts = dict(tp=int(((labels==1)&pred).sum()), fp=int(((labels==0)&pred).sum()),
                  fn=int(((labels==1)&~pred).sum()), tn=int(((labels==0)&~pred).sum()))
    assert counts == {k: selected['metrics']['fit']['windows'][k] for k in counts}
    by_row = defaultdict(list)
    for i,w in enumerate(windows):
        assert w['partition'] == 'fit' and w['response_id'] in rows
        by_row[w['response_id']].append((i,w))
    candidates = []; stats = defaultdict(Counter)
    for t in tokens:
        rid = t['response_id']; row = rows[rid]
        assert t['original_response'] == row['original_response']
        assert t['original_labels'] == row['labels']
        for mapping in t['span_token_mapping']:
            si = mapping['span_index']; gold = t['original_labels'][si]
            assert row['original_response'][gold['start']:gold['end']] == gold['text']
            risk = set(mapping['risk_token_indices'])
            overlap = [i for i,w in by_row[rid] if risk.intersection(w['token_indices'])]
            interior = [i for i,w in by_row[rid]
                        if len(w['token_indices']) == 4 and set(w['token_indices']) <= risk]
            kind = gold['label_type']; stats[kind]['all_original_spans'] += 1
            if overlap and not pred[overlap].any():
                stats[kind]['completely_missed_spans'] += 1
                stats[kind]['missed_with_four_risk_token_interior'] += bool(interior)
                candidates.append(dict(response_id=rid, source_id=row['source_id'], span_index=si,
                    original_gold=gold, risk_token_indices=sorted(risk),
                    overlap_window_indices=overlap, interior_window_indices=interior,
                    overlap_window_count=len(overlap), interior_window_count=len(interior),
                    max_overlapping_window_score=float(scores[overlap].max()),
                    source_jsonl_line=line_numbers[rid], question=row['question'],
                    retrieved_passages=row['retrieved_passages'], original_response=row['original_response'],
                    all_original_labels=row['labels'], answer_sha256=row['answer_sha256'],
                    answer_alert_at_original_answer_threshold=bool(scores[[i for i,_ in by_row[rid]]].max()
                        >= selected['thresholds']['answer']['threshold'])))
    cases=[]
    for kind in KINDS:
        cc = sorted([c for c in candidates if c['original_gold']['label_type'] == kind],
                    key=lambda c:(not bool(c['interior_window_count']), int(c['response_id']),
                                  c['original_gold']['start'], c['span_index']))[:4]
        for c in cc:
            c['case_id'] = f"{c['response_id']}__span{c['span_index']}"
        cases.extend(cc)
    assert len(cases) == 12 and all(c['interior_window_count'] for c in cases)
    result = dict(status='completed', model_candidate=selected['candidate'], threshold=threshold,
        original_answer_threshold=selected['thresholds']['answer']['threshold'],
        selection_rule='Per requested type: prefer missed spans containing a full four-risk-token raw-BPE window; numeric response_id then original char start, first four. Completely missed means zero alerts among ALL windows intersecting this span risk tokens. Scores are not used to rank cases.',
        fit_answers=634, fit_windows=len(windows), fit_counts=counts, counts_by_type=dict(stats),
        cases=cases, unique_selected_answers=len({c['response_id'] for c in cases}),
        provenance={str(p.relative_to(ROOT)):sha(p) for p in paths+[MODEL/'summary.json',score_path]},
        no_calibration_text_or_labels_read=True, no_official_test=True, no_model_inference=True,
        no_training=True, labels_unchanged=True,
        limitation='Purposive missed-span cases cannot estimate dataset noise or population failure prevalence. Human labels remain canonical; later explanations are assistant interpretations.')
    (OUT/'CASES.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'completed','cases':len(cases),'answers':result['unique_selected_answers'],
                      'counts':dict(stats),'path':str(OUT/'CASES.json')},ensure_ascii=False))


if __name__ == '__main__':
    run()
